#!/usr/bin/env python3
"""Colab VM orchestrator for unattended MTool -> AiNiee 7.3 -> Google Drive translation.

Runs inside a Colab GPU session (recommended: G4 RTX PRO 6000 96GB):
1. Bootstraps isolated uv virtualenvs for vLLM and AiNiee backend.
2. Starts local vLLM server with Blackwell FP8 + Prefix Caching + thinking disabled.
3. Executes ainiee_runner.py (terminology extraction + translation + project save/export).
4. Periodically backs up AiNiee project cache to Google Drive during long jobs.
5. Uploads final deliverable (official AiNiee project + exported MTool files) to Google Drive.
6. Cleans temporary credentials, stops vLLM, and optionally self-terminates Colab VM.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tarfile
import threading
import time
import urllib.request
from pathlib import Path
from typing import Any


MODEL_PRESETS: dict[str, dict[str, Any]] = {
    # Best overall quality + 18x uncensored + native Blackwell FP8 (~28.8 GB VRAM, ~65 GB free KV cache on G4)
    "quality": {
        "model": "munekazu/Huihui-Qwen3.8-27B-abliterated-FP8",
        "max_model_len": 32768,
        "gpu_memory_utilization": 0.90,
        "threads": 24,
        "reasoning_parser": "qwen3",
        "disable_thinking": True,
        "language_model_only": True,
        "extra_vllm_args": [],
    },
    # Fastest MoE throughput (35.9B total, 3B active, ~35.8 GB FP8) - lowest CU per 100k words
    "fast": {
        "model": "coolthor/Huihui-Qwen3.6-35B-A3B-abliterated-FP8-DYNAMIC",
        "max_model_len": 32768,
        "gpu_memory_utilization": 0.90,
        "threads": 32,
        "reasoning_parser": "qwen3",
        "disable_thinking": True,
        "language_model_only": True,
        "extra_vllm_args": [],
    },
    # Full BF16 Qwen3.8-27B abliterated (~55.5 GB VRAM on G4 96GB) - zero quantization loss
    "bf16": {
        "model": "huihui-ai/Huihui-Qwen3.8-27B-abliterated",
        "max_model_len": 32768,
        "gpu_memory_utilization": 0.92,
        "threads": 16,
        "reasoning_parser": "qwen3",
        "disable_thinking": True,
        "language_model_only": True,
        "extra_vllm_args": [],
    },
}

# Lightweight headless deps required for AiNiee 7.3 MTool extraction + translation + export.
AINIEE_DEPS = [
    "fastapi>=0.115,<1",
    "uvicorn[standard]>=0.30,<1",
    "openai>=2.46.0",
    "tiktoken>=0.13.0",
    "openpyxl>=3.1.5",
    "PyYAML>=6.0.3",
    "requests>=2.34.2",
    "httpx[http2]>=0.28.1",
    "rich>=15.0.0",
    "tqdm>=4.69.0",
    "jaconv>=0.5.0",
    "python-rapidjson>=1.23",
    "language-data>=1.4.0",
    "langcodes>=3.5.1",
    "msgspec>=0.21.1",
    "chardet>=7.4.3",
    "charset-normalizer>=3.0.0",
    "pydantic>=2.10.6",
    "regex>=2024.11.6",
    "orjson>=3.10.14",
    "tenacity>=9.0.0",
    "psutil>=7.0.0",
    "beautifulsoup4>=4.15.0",
    "lxml>=6.1.1",
    "Pillow>=10.4",
    "levenshtein>=0.27.1",
    "pandas>=2.2.0",
    "cryptography>=44.0.0",
    "anthropic>=0.40.0",
    "google-genai>=1.0.0",
    "boto3>=1.35.0",
    "mcp>=1.0.0",
    "curl_cffi>=0.10.0",
    "magika>=0.6.0",
    "polib>=1.2.0",
    "python-pptx>=1.0.0",
    # MediaPipe language detector deps (mediapipe itself is installed with --no-deps per AiNiee official spec)
    "protobuf==4.25.7",
    "absl-py>=2.0.0",
    "attrs>=19.1.0",
    "flatbuffers>=2.0",
    "matplotlib>=3.8.0",
    "opencv-python-headless>=4.10.0",
    "onnx>=1.16.0",
    "onnxruntime>=1.19.0",
]


def log(msg: str) -> None:
    ts = time.strftime("%H:%M:%S")
    print(f"[Worker {ts}] {msg}", flush=True)


def run_cmd(cmd: list[str], *, env: dict[str, str] | None = None, check: bool = True) -> subprocess.CompletedProcess:
    log("$ " + " ".join(cmd))
    return subprocess.run(cmd, env=env, check=check)


def ensure_uv() -> str:
    uv_path = shutil.which("uv")
    if uv_path:
        return uv_path
    log("Installing uv...")
    run_cmd([sys.executable, "-m", "pip", "install", "-q", "uv"])
    uv_path = shutil.which("uv") or str(Path(sys.executable).parent / "uv")
    return uv_path


def ensure_rclone() -> str:
    rclone_path = shutil.which("rclone")
    if rclone_path:
        return rclone_path
    log("Installing rclone for Google Drive upload...")
    run_cmd(["bash", "-c", "apt-get update -qq && apt-get install -y -qq rclone >/dev/null 2>&1 || curl -fsSL https://rclone.org/install.sh | bash"])
    rclone_path = shutil.which("rclone")
    if not rclone_path:
        raise RuntimeError("Failed to install rclone")
    return rclone_path


def bootstrap_vllm_env(uv: str, venv_vllm: Path) -> None:
    """Create isolated Python 3.12 virtualenv for vLLM + huggingface_hub (Xet)."""
    vllm_py = venv_vllm / "bin" / "python"
    if not (venv_vllm / "bin" / "vllm").exists():
        log("Setting up vLLM virtualenv (Python 3.12 + vllm + huggingface_hub)...")
        run_cmd([uv, "venv", "--python", "3.12", str(venv_vllm)])
        run_cmd([uv, "pip", "install", "--python", str(vllm_py), "vllm", "huggingface_hub"])


def bootstrap_ainiee_env(uv: str, ainiee_dir: Path, ainiee_ref: str, venv_ainiee: Path) -> None:
    """Clone AiNiee and install headless backend deps (runs concurrently with vLLM startup)."""
    if not (ainiee_dir / "Backend").is_dir():
        log(f"Cloning NEKOparapa/AiNiee (ref={ainiee_ref})...")
        if ainiee_dir.exists():
            shutil.rmtree(ainiee_dir)
        run_cmd(
            [
                "git",
                "clone",
                "--depth",
                "1",
                "--branch",
                ainiee_ref,
                "https://github.com/NEKOparapa/AiNiee.git",
                str(ainiee_dir),
            ]
        )

    ainiee_py = venv_ainiee / "bin" / "python"
    marker = venv_ainiee / ".deps_ready"
    if not marker.exists():
        log("Setting up AiNiee headless virtualenv (Python 3.12)...")
        run_cmd([uv, "venv", "--python", "3.12", str(venv_ainiee)])
        run_cmd([uv, "pip", "install", "--python", str(ainiee_py), *AINIEE_DEPS])
        # Official AiNiee spec installs mediapipe with --no-deps to avoid protobuf/opencv conflicts
        run_cmd([uv, "pip", "install", "--python", str(ainiee_py), "--no-deps", "mediapipe==0.10.14"])
        marker.write_text("ok", encoding="utf-8")


def start_vllm(
    venv_vllm: Path,
    model_id: str,
    max_model_len: int,
    gpu_mem: float,
    reasoning_parser: str,
    disable_thinking: bool,
    language_model_only: bool,
    extra_args: list[str],
    log_path: Path,
    hf_token: str = "",
) -> subprocess.Popen:
    vllm_bin = str(venv_vllm / "bin" / "vllm")
    cmd = [
        vllm_bin,
        "serve",
        model_id,
        "--host",
        "127.0.0.1",
        "--port",
        "8000",
        "--served-model-name",
        "translator",
        "--tensor-parallel-size",
        "1",
        "--max-model-len",
        str(max_model_len),
        "--gpu-memory-utilization",
        str(gpu_mem),
        "--enable-prefix-caching",
        "--disable-flashinfer-autotune",
        "--trust-remote-code",
    ]
    if language_model_only:
        cmd.append("--language-model-only")
    if reasoning_parser:
        cmd.extend(["--reasoning-parser", reasoning_parser])
    if disable_thinking:
        cmd.extend(["--default-chat-template-kwargs", '{"enable_thinking": false}'])
    cmd.extend(extra_args)

    env = os.environ.copy()
    env["PATH"] = f"{venv_vllm / 'bin'}:/usr/local/cuda/bin:{env.get('PATH', '')}"
    if Path("/usr/local/cuda").is_dir():
        env.setdefault("CUDA_HOME", "/usr/local/cuda")
    env["HF_XET_HIGH_PERFORMANCE"] = "1"
    env["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
    env["VLLM_USE_DEEP_GEMM"] = "0"
    env["PYTHONUNBUFFERED"] = "1"
    if hf_token:
        env["HF_TOKEN"] = hf_token

    log(f"Starting vLLM: {' '.join(cmd)}")
    log_file = open(log_path, "w", encoding="utf-8")
    proc = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT, env=env)
    proc._log_file = log_file  # type: ignore[attr-defined]
    return proc


def wait_for_vllm(proc: subprocess.Popen, log_path: Path, timeout_sec: int = 1200) -> None:
    deadline = time.time() + timeout_sec
    last_report = 0.0
    while time.time() < deadline:
        rc = proc.poll()
        if rc is not None:
            tail = log_path.read_text(encoding="utf-8", errors="replace")[-4000:]
            raise RuntimeError(f"vLLM exited prematurely (rc={rc}). Tail of log:\n{tail}")
        try:
            with urllib.request.urlopen("http://127.0.0.1:8000/v1/models", timeout=3) as resp:
                if resp.status == 200:
                    data = json.loads(resp.read().decode("utf-8"))
                    log(f"vLLM READY! Served models: {[m.get('id') for m in data.get('data', [])]}")
                    return
        except Exception:
            pass
        now = time.time()
        if now - last_report >= 20.0:
            lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
            hint = lines[-1] if lines else "starting..."
            log(f"Waiting for vLLM startup... ({hint[:140]})")
            last_report = now
        time.sleep(3)
    raise TimeoutError(f"vLLM did not become ready within {timeout_sec}s")


def stop_process(proc: subprocess.Popen | None) -> None:
    if proc is None:
        return
    try:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                proc.kill()
    finally:
        lf = getattr(proc, "_log_file", None)
        if lf:
            try:
                lf.close()
            except Exception:
                pass


def upload_directory_to_gdrive(
    local_dir: Path,
    gdrive_mode: str,
    gdrive_dir: str,
    run_name: str,
    rclone_conf: str,
    rclone_remote: str,
    is_checkpoint: bool = False,
) -> str | None:
    if gdrive_mode == "none" or not local_dir.exists():
        return None

    rel_target = f"{gdrive_dir.strip('/')}/{run_name}"
    if is_checkpoint:
        rel_target = f"{rel_target}/_checkpoint"

    if gdrive_mode == "drivemount":
        mount_root = Path("/content/drive/MyDrive")
        if not mount_root.is_dir():
            log("WARNING: /content/drive/MyDrive not mounted; skipping drivemount copy.")
            return None
        dest = mount_root / rel_target
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copytree(local_dir, dest, dirs_exist_ok=True)
        subprocess.run(["sync"], check=False)
        log(f"Saved to Google Drive (mount): {dest}")
        return str(dest)

    if gdrive_mode == "rclone":
        if not rclone_conf or not Path(rclone_conf).is_file():
            log("WARNING: rclone.conf missing; skipping rclone upload.")
            return None
        rclone_bin = ensure_rclone()
        remote_target = f"{rclone_remote.rstrip(':')}:{rel_target}"
        run_cmd(
            [
                rclone_bin,
                "--config",
                rclone_conf,
                "copy",
                str(local_dir),
                remote_target,
                "--transfers",
                "4",
                "--checkers",
                "4",
                "--low-level-retries",
                "10",
                "--retries",
                "5",
                "--tpslimit",
                "8",
            ]
        )
        log(f"Uploaded to Google Drive (rclone): {remote_target}")
        return remote_target

    return None


def start_checkpoint_thread(
    stop_event: threading.Event,
    project_meta_file: Path,
    interval_min: int,
    gdrive_mode: str,
    gdrive_dir: str,
    run_name: str,
    rclone_conf: str,
    rclone_remote: str,
) -> threading.Thread | None:
    if gdrive_mode == "none" or interval_min <= 0:
        return None

    def loop() -> None:
        while not stop_event.wait(interval_min * 60):
            try:
                if not project_meta_file.is_file():
                    continue
                meta = json.loads(project_meta_file.read_text(encoding="utf-8"))
                cache_dir = Path(meta.get("project_cache_dir", ""))
                if cache_dir.is_dir():
                    log(f"Periodic checkpoint backup -> Google Drive ({cache_dir.name})")
                    upload_directory_to_gdrive(
                        cache_dir,
                        gdrive_mode,
                        gdrive_dir,
                        run_name,
                        rclone_conf,
                        rclone_remote,
                        is_checkpoint=True,
                    )
            except Exception as exc:
                log(f"Checkpoint backup warning (non-fatal): {exc}")

    t = threading.Thread(target=loop, name="gdrive-checkpoint", daemon=True)
    t.start()
    return t


def build_deliverable(
    deliverable_dir: Path,
    run_name: str,
    output_dir: Path,
    glossary_json: Path,
    ainiee_result: dict[str, Any],
    vllm_log: Path,
    report: dict[str, Any],
    worker_log: Path | None = None,
) -> Path:
    if deliverable_dir.exists():
        shutil.rmtree(deliverable_dir)
    deliverable_dir.mkdir(parents=True, exist_ok=True)

    # 1. Translated MTool output files
    if output_dir.is_dir():
        shutil.copytree(output_dir, deliverable_dir / "output", dirs_exist_ok=True)

    # 2. Official AiNiee 7.3 project folder (can be opened directly in desktop AiNiee GUI)
    proj_root = deliverable_dir / "AiNiee_Project"
    proj_root.mkdir(parents=True, exist_ok=True)
    cache_dir_str = ainiee_result.get("project_cache_dir", "")
    project_id = ainiee_result.get("project_id", "")
    if cache_dir_str and Path(cache_dir_str).is_dir():
        target_cache = proj_root / "ProjectCache" / (project_id or Path(cache_dir_str).name)
        shutil.copytree(cache_dir_str, target_cache, dirs_exist_ok=True)
        # Also place a top-level copy of AinieeCacheData.json for quick one-click "打开缓存文件" in AiNiee
        top_cache = Path(cache_dir_str) / "AinieeCacheData.json"
        if top_cache.is_file():
            shutil.copy2(top_cache, proj_root / "AinieeCacheData.json")

    cfg_str = ainiee_result.get("config_file", "")
    if cfg_str and Path(cfg_str).is_file():
        shutil.copy2(cfg_str, proj_root / "config.json")

    # 3. Standalone human-readable extracted glossary
    if glossary_json.is_file():
        shutil.copy2(glossary_json, deliverable_dir / "extracted_glossary.json")

    # 4. Run report & tail of vLLM log
    (deliverable_dir / "run_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if vllm_log.is_file():
        tail = vllm_log.read_text(encoding="utf-8", errors="replace")[-30000:]
        (deliverable_dir / "vllm_tail.log").write_text(tail, encoding="utf-8")
    if worker_log is not None and worker_log.is_file():
        shutil.copy2(worker_log, deliverable_dir / "worker.log")

    # 5. Single tar.gz archive for easy local/Drive download
    archive_path = deliverable_dir.parent / f"{run_name}.tar.gz"
    with tarfile.open(archive_path, "w:gz") as tar:
        tar.add(deliverable_dir, arcname=run_name)
    shutil.copy2(archive_path, deliverable_dir / f"{run_name}.tar.gz")
    return archive_path


def self_unassign_colab() -> None:
    log("Self-terminating Colab runtime via google.colab.runtime.unassign()...")
    try:
        from google.colab import runtime as colab_runtime  # type: ignore

        colab_runtime.unassign()
    except Exception as exc:
        log(f"runtime.unassign() note: {exc} (local launcher will also call colab stop)")


def main() -> int:
    parser = argparse.ArgumentParser(description="Unattended Colab MTool -> AiNiee worker")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--input-dir", default="/content/work/input")
    parser.add_argument("--work-dir", default="/content/work")
    parser.add_argument("--ainiee-dir", default="/content/AiNiee")
    parser.add_argument("--ainiee-ref", default="AiNiee7.3.0")
    parser.add_argument("--model-preset", choices=sorted(MODEL_PRESETS.keys()), default="quality")
    parser.add_argument("--model-id", default="", help="Override HF model repo ID")
    parser.add_argument("--threads", type=int, default=0, help="Override AiNiee concurrency (0 = preset default)")
    parser.add_argument("--max-model-len", type=int, default=0)
    parser.add_argument("--tokens-limit", type=int, default=1024)
    parser.add_argument("--extract-token-limit", type=int, default=8000)
    parser.add_argument("--pre-line-counts", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.35)
    parser.add_argument("--skip-extract", action="store_true")
    parser.add_argument("--language-filter", action="store_true")
    parser.add_argument("--code-filter", action="store_true")
    parser.add_argument("--resume-cache", default="")
    parser.add_argument("--gdrive-mode", choices=["none", "rclone", "drivemount"], default="none")
    parser.add_argument("--gdrive-dir", default="AiNiee_Translations")
    parser.add_argument("--rclone-conf", default="/tmp/rclone.conf")
    parser.add_argument("--rclone-remote", default="gdrive")
    parser.add_argument("--checkpoint-minutes", type=int, default=10)
    parser.add_argument("--no-language-model-only", action="store_true")
    parser.add_argument("--auto-unassign", action="store_true")
    args = parser.parse_args()

    preset = MODEL_PRESETS[args.model_preset]
    model_id = args.model_id or preset["model"]
    max_model_len = args.max_model_len or preset["max_model_len"]
    threads = args.threads or preset["threads"]
    use_lm_only = bool(preset.get("language_model_only", True) and not args.no_language_model_only)

    work_dir = Path(args.work_dir).resolve()
    input_dir = Path(args.input_dir).resolve()
    output_dir = work_dir / "output"
    deliverable_root = work_dir / "deliverable"
    deliverable_dir = deliverable_root / args.run_name
    ainiee_dir = Path(args.ainiee_dir).resolve()
    venv_vllm = Path("/content/venv-vllm")
    venv_ainiee = Path("/content/venv-ainiee")
    vllm_log = work_dir / "vllm.log"
    worker_log = work_dir / "worker.log"
    worker_status_file = work_dir / "worker.status"
    result_json = work_dir / "ainiee_result.json"
    glossary_json = work_dir / "extracted_glossary.json"
    project_meta_file = work_dir / "project_meta.json"
    ainiee_runner_path = Path("/content/ainiee_runner.py")

    work_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    worker_status_file.write_text("RUNNING\n", encoding="utf-8")

    t_start = time.time()
    vllm_proc: subprocess.Popen | None = None
    ainiee_boot_thread: threading.Thread | None = None
    checkpoint_stop = threading.Event()
    checkpoint_thread: threading.Thread | None = None
    ainiee_result: dict[str, Any] = {}
    exit_code = 1
    report: dict[str, Any] = {
        "run_name": args.run_name,
        "model_preset": args.model_preset,
        "model_id": model_id,
        "threads": threads,
        "language_model_only": use_lm_only,
        "gdrive_mode": args.gdrive_mode,
        "gdrive_destination": None,
        "status": "failed",
        "error": None,
    }

    try:
        # Print GPU info
        gpu_info = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total,compute_cap", "--format=csv,noheader"],
            text=True,
        ).strip()
        report["gpu"] = gpu_info
        log(f"GPU detected: {gpu_info}")
        log(f"Preset={args.model_preset} | Model={model_id} | Threads={threads} | LM_only={use_lm_only}")

        uv = ensure_uv()

        # 1. Install vLLM env first so weight download + server startup begin immediately
        t_vllm_boot = time.time()
        bootstrap_vllm_env(uv, venv_vllm)
        report["vllm_bootstrap_seconds"] = round(time.time() - t_vllm_boot, 2)
        log(f"vLLM environment ready in {report['vllm_bootstrap_seconds']}s")

        # 2. Start vLLM & concurrently bootstrap AiNiee headless env in the background
        t_vllm = time.time()
        vllm_proc = start_vllm(
            venv_vllm=venv_vllm,
            model_id=model_id,
            max_model_len=max_model_len,
            gpu_mem=preset["gpu_memory_utilization"],
            reasoning_parser=preset["reasoning_parser"],
            disable_thinking=preset["disable_thinking"],
            language_model_only=use_lm_only,
            extra_args=preset["extra_vllm_args"],
            log_path=vllm_log,
            hf_token=os.environ.get("HF_TOKEN", ""),
        )

        ainiee_boot_state: dict[str, Any] = {"seconds": 0.0, "error": None}

        def _boot_ainiee() -> None:
            t_ab = time.time()
            try:
                bootstrap_ainiee_env(uv, ainiee_dir, args.ainiee_ref, venv_ainiee)
            except Exception as exc:
                ainiee_boot_state["error"] = exc
            finally:
                ainiee_boot_state["seconds"] = round(time.time() - t_ab, 2)

        ainiee_boot_thread = threading.Thread(target=_boot_ainiee, name="ainiee-bootstrap", daemon=True)
        ainiee_boot_thread.start()

        wait_for_vllm(vllm_proc, vllm_log)
        report["vllm_startup_seconds"] = round(time.time() - t_vllm, 2)

        ainiee_boot_thread.join()
        report["ainiee_bootstrap_seconds"] = ainiee_boot_state["seconds"]
        log(f"AiNiee environment ready in {report['ainiee_bootstrap_seconds']}s (overlapped with vLLM)")
        if ainiee_boot_state["error"] is not None:
            raise ainiee_boot_state["error"]

        # Start periodic checkpoint backup to Google Drive
        checkpoint_thread = start_checkpoint_thread(
            stop_event=checkpoint_stop,
            project_meta_file=project_meta_file,
            interval_min=args.checkpoint_minutes,
            gdrive_mode=args.gdrive_mode,
            gdrive_dir=args.gdrive_dir,
            run_name=args.run_name,
            rclone_conf=args.rclone_conf,
            rclone_remote=args.rclone_remote,
        )

        # Execute headless AiNiee 7.3 pipeline
        t_ainiee = time.time()
        runner_cmd = [
            str(venv_ainiee / "bin" / "python"),
            str(ainiee_runner_path),
            "--ainiee-dir",
            str(ainiee_dir),
            "--input-dir",
            str(input_dir),
            "--output-dir",
            str(output_dir),
            "--result-json",
            str(result_json),
            "--glossary-json",
            str(glossary_json),
            "--project-meta-file",
            str(project_meta_file),
            "--api-url",
            "http://127.0.0.1:8000/v1",
            "--model-name",
            "translator",
            "--threads",
            str(threads),
            "--tokens-limit",
            str(args.tokens_limit),
            "--extract-token-limit",
            str(args.extract_token_limit),
            "--pre-line-counts",
            str(args.pre_line_counts),
            "--temperature",
            str(args.temperature),
        ]
        if args.skip_extract:
            runner_cmd.append("--skip-extract")
        if args.language_filter:
            runner_cmd.append("--language-filter")
        if args.code_filter:
            runner_cmd.append("--code-filter")
        if args.resume_cache:
            runner_cmd.extend(["--resume-cache", args.resume_cache])

        ainiee_env = os.environ.copy()
        ainiee_env["MPLBACKEND"] = "Agg"
        ainiee_env["PYTHONUNBUFFERED"] = "1"
        proc_ainiee = subprocess.run(runner_cmd, env=ainiee_env, check=False)
        report["ainiee_total_seconds"] = round(time.time() - t_ainiee, 2)
        if result_json.is_file():
            ainiee_result = json.loads(result_json.read_text(encoding="utf-8"))
        report["ainiee"] = ainiee_result
        exit_code = proc_ainiee.returncode
        report["status"] = "succeeded" if exit_code == 0 else "failed"

    except Exception as exc:
        report["status"] = "failed"
        report["error"] = repr(exc)
        log(f"ERROR: {exc!r}")
        exit_code = 1
    finally:
        checkpoint_stop.set()
        if ainiee_boot_thread is not None and ainiee_boot_thread.is_alive():
            ainiee_boot_thread.join(timeout=5)
        if checkpoint_thread is not None:
            checkpoint_thread.join(timeout=5)
        stop_process(vllm_proc)
        report["total_seconds"] = round(time.time() - t_start, 2)

        # Always package and save whatever succeeded (even partial progress!)
        try:
            if not ainiee_result and project_meta_file.is_file():
                meta = json.loads(project_meta_file.read_text(encoding="utf-8"))
                ainiee_result.update(meta)
            archive_path = build_deliverable(
                deliverable_dir=deliverable_dir,
                run_name=args.run_name,
                output_dir=output_dir,
                glossary_json=glossary_json,
                ainiee_result=ainiee_result,
                vllm_log=vllm_log,
                report=report,
                worker_log=worker_log,
            )
            log(f"Deliverable packaged: {archive_path} ({archive_path.stat().st_size / 1024:.1f} KB)")

            dest = upload_directory_to_gdrive(
                local_dir=deliverable_dir,
                gdrive_mode=args.gdrive_mode,
                gdrive_dir=args.gdrive_dir,
                run_name=args.run_name,
                rclone_conf=args.rclone_conf,
                rclone_remote=args.rclone_remote,
                is_checkpoint=False,
            )
            report["gdrive_destination"] = dest
            (deliverable_dir / "run_report.json").write_text(
                json.dumps(report, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception as save_exc:
            log(f"WARNING: deliverable packaging/upload failed: {save_exc!r}")

        # Wipe temporary rclone credentials from /tmp
        if args.rclone_conf and Path(args.rclone_conf).is_file():
            try:
                Path(args.rclone_conf).unlink()
            except Exception:
                pass

        try:
            worker_status_file.write_text(
                "SUCCEEDED\n" if report["status"] == "succeeded" else "FAILED\n",
                encoding="utf-8",
            )
        except Exception:
            pass

        log(f"FINAL SUMMARY: status={report['status']} | total={report['total_seconds']}s | gdrive={report.get('gdrive_destination')}")
        if args.auto_unassign:
            self_unassign_colab()

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
