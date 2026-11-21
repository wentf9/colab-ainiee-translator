#!/usr/bin/env python3
"""One-click unattended MTool -> AiNiee 7.3 -> Google Drive translator on Google Colab.

Example usage:
  # 100% zero-click unattended (after one-time ./setup_gdrive.sh)
  python3 run_colab_translate.py ./game/ManualTransFile.json --game "MyGalgame"

  # Using native Colab Drive mount (one browser click at start, then unattended)
  python3 run_colab_translate.py ./game/ManualTransFile.json --game "MyGalgame" --gdrive-mode drivemount

  # Local download only (no Google Drive needed, great for testing)
  python3 run_colab_translate.py ./sample_mtool/ManualTransFile.json --game "SmokeTest" --gdrive-mode local
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent


def ensure_local_bin_in_path() -> None:
    """Prepend ~/.local/bin so non-login SSH shells always resolve colab and rclone."""
    local_bin = str(Path.home() / ".local" / "bin")
    current = os.environ.get("PATH", "")
    parts = current.split(os.pathsep) if current else []
    if local_bin not in parts:
        os.environ["PATH"] = f"{local_bin}{os.pathsep}{current}" if current else local_bin


def load_dotenv_value(env_path: Path, key: str) -> str:
    """Read a single KEY=VALUE entry from a .env file if present."""
    if not env_path.is_file():
        return ""
    for raw_line in env_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        if k.strip() != key:
            continue
        val = v.strip()
        if len(val) >= 2 and val[0] == val[-1] and val[0] in ("'", '"'):
            val = val[1:-1]
        return val
    return ""


def upload_file_auto_gz(
    session_name: str,
    local_path: Path,
    remote_path: str,
    threshold_bytes: int = 2 * 1024 * 1024,
) -> None:
    """Upload small files directly; gzip-compress larger files to avoid Colab HTTP 413."""
    size = local_path.stat().st_size
    if size <= threshold_bytes:
        run_colab(["upload", "-s", session_name, str(local_path), remote_path])
        return

    tmp_gz_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".gz", delete=False) as tmp:
            tmp_gz_path = Path(tmp.name)
        with open(local_path, "rb") as fin, gzip.open(tmp_gz_path, "wb", compresslevel=6) as fout:
            shutil.copyfileobj(fin, fout)
        gz_size = tmp_gz_path.stat().st_size
        log(
            f"大文件透明压缩上传: {local_path.name} "
            f"({size / (1024 * 1024):.2f} MB -> {gz_size / (1024 * 1024):.2f} MB)"
        )
        remote_gz = remote_path + ".gz"
        run_colab(["upload", "-s", session_name, str(tmp_gz_path), remote_gz])
        decompress_code = (
            "import gzip, os, shutil\n"
            f"remote_gz = {json.dumps(remote_gz)}\n"
            f"remote_path = {json.dumps(remote_path)}\n"
            "with gzip.open(remote_gz, 'rb') as fin, open(remote_path, 'wb') as fout:\n"
            "    shutil.copyfileobj(fin, fout)\n"
            "os.remove(remote_gz)\n"
        )
        subprocess.run(
            ["colab", "exec", "-s", session_name, "--timeout", "120"],
            input=decompress_code,
            text=True,
            check=True,
        )
    finally:
        if tmp_gz_path is not None and tmp_gz_path.exists():
            tmp_gz_path.unlink()


def log(msg: str) -> None:
    ts = time.strftime("%H:%M:%S")
    print(f"[Launcher {ts}] {msg}", flush=True)


def sanitize_name(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9\u4e00-\u9fff._-]+", "_", name).strip("_")
    return cleaned or "game"


def run_colab(args: list[str], *, check: bool = True, capture: bool = False) -> subprocess.CompletedProcess:
    cmd = ["colab", *args]
    if not capture:
        log("$ " + " ".join(cmd))
    return subprocess.run(cmd, check=check, text=True, capture_output=capture)


def resolve_gdrive_mode(requested: str, rclone_conf: Path) -> tuple[str, Path | None]:
    if requested == "local":
        return "none", None
    if requested == "drivemount":
        return "drivemount", None
    if requested == "rclone":
        if not rclone_conf.is_file():
            raise FileNotFoundError(
                f"找不到 rclone 配置文件: {rclone_conf}\n"
                "请先运行一次 ./setup_gdrive.sh 完成 Google Drive 授权，或改用 --gdrive-mode drivemount / local"
            )
        return "rclone", rclone_conf
    # auto
    if rclone_conf.is_file():
        log(f"检测到 {rclone_conf}，自动使用 rclone 零点击无人值守上传 Google Drive。")
        return "rclone", rclone_conf
    log(
        f"未检测到 {rclone_conf}。如需真正零点击上传 Google Drive，请先运行 ./setup_gdrive.sh。\n"
        "当前自动切换为 native colab drivemount 模式（启动时需点击一次浏览器授权，之后全程无人值守）。"
    )
    return "drivemount", None


def collect_input_files(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    if input_path.is_dir():
        files = sorted(p for p in input_path.glob("*.json") if p.is_file())
        if not files:
            raise ValueError(f"输入目录下未找到任何 .json 文件: {input_path}")
        return files
    raise FileNotFoundError(f"输入路径不存在: {input_path}")


def monitor_detached_colab_session(
    session_name: str,
    already_printed_bytes: int,
    poll_sec: int = 60,
) -> str:
    """Poll Colab status & worker.log/worker.status if colab exec WebSocket drops mid-run."""
    log(
        f"检测到 colab exec 连接已中断，切换至断点轮询守护模式（每 {poll_sec}s 拉取一次远程 worker.log / worker.status）..."
    )
    consecutive_idle = 0
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_dir = Path(tmpdir)
        local_wlog = tmp_dir / "worker.log"
        local_wstatus = tmp_dir / "worker.status"
        while True:
            st = run_colab(["status", "-s", session_name], check=False, capture=True)
            st_out = f"{st.stdout or ''}\n{st.stderr or ''}".strip()
            st_lower = st_out.lower()
            if st.returncode != 0 or any(
                k in st_lower for k in ("not found", "no active session", "terminated", "404")
            ):
                log("Colab 会话已自行结束（已完成云盘保存并触发 runtime.unassign）")
                return "unassigned"

            dl_log = run_colab(
                ["download", "-s", session_name, "/content/work/worker.log", str(local_wlog)],
                check=False,
                capture=True,
            )
            if dl_log.returncode == 0 and local_wlog.is_file():
                raw = local_wlog.read_bytes()
                if len(raw) > already_printed_bytes:
                    chunk = raw[already_printed_bytes:].decode("utf-8", errors="replace")
                    sys.stdout.write(chunk)
                    sys.stdout.flush()
                    already_printed_bytes = len(raw)

            worker_status = ""
            dl_status = run_colab(
                ["download", "-s", session_name, "/content/work/worker.status", str(local_wstatus)],
                check=False,
                capture=True,
            )
            if dl_status.returncode == 0 and local_wstatus.is_file():
                worker_status = local_wstatus.read_text(encoding="utf-8", errors="replace").strip()
                if worker_status in ("SUCCEEDED", "FAILED"):
                    log(f"轮询检测到远程任务终态: worker.status={worker_status}")
                    return worker_status

            if "status: idle" in st_lower and worker_status != "RUNNING":
                consecutive_idle += 1
                if consecutive_idle >= 2:
                    log("轮询检测到 Colab 内核已连续两次 IDLE 且非 RUNNING 状态，结束等待。")
                    return "idle_ended"
            else:
                consecutive_idle = 0

            time.sleep(poll_sec)


def main() -> int:
    ensure_local_bin_in_path()
    parser = argparse.ArgumentParser(
        description="One-click Colab G4/A100 MTool -> AiNiee 7.3 -> Google Drive unattended pipeline"
    )
    parser.add_argument("input_path", help="Path to MTool ManualTransFile.json or folder of JSON files")
    parser.add_argument("--game", default="", help="Project / game name (used for Drive folder name)")
    parser.add_argument("--gpu", default="G4", choices=["G4", "A100", "L4", "T4"], help="Colab GPU (default: G4 96GB)")
    parser.add_argument(
        "--model-preset",
        default="quality",
        choices=["quality", "fast", "bf16"],
        help="quality=Qwen3.8-27B-abliterated-FP8 (default), fast=Qwen3.6-35B-A3B-FP8, bf16=Qwen3.8-27B-BF16",
    )
    parser.add_argument("--model-id", default="", help="Optional custom HuggingFace model ID")
    parser.add_argument("--threads", type=int, default=0, help="AiNiee concurrency (0 = preset default)")
    parser.add_argument("--tokens-limit", type=int, default=1024)
    parser.add_argument("--extract-token-limit", type=int, default=8000)
    parser.add_argument("--pre-line-counts", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.35)
    parser.add_argument("--skip-extract", action="store_true", help="Skip AiNiee terminology extraction")
    parser.add_argument("--language-filter", action="store_true", help="Enable AiNiee language_filter_switch (skip pure ASCII/non-source-lang)")
    parser.add_argument("--code-filter", action="store_true", help="Enable AiNiee code_filter_switch (Magika)")
    parser.add_argument("--resume-cache", default="", help="Optional local AinieeCacheData.json to resume translation")
    parser.add_argument(
        "--gdrive-mode",
        default="auto",
        choices=["auto", "rclone", "drivemount", "local"],
        help="Google Drive save mode: auto / rclone (zero-click) / drivemount / local (download only)",
    )
    parser.add_argument("--gdrive-dir", default="AiNiee_Translations", help="Destination folder in Google Drive")
    parser.add_argument("--rclone-conf", default=str(Path.home() / ".config" / "rclone" / "rclone.conf"))
    parser.add_argument("--rclone-remote", default="gdrive", help="rclone remote name (default: gdrive)")
    parser.add_argument("--checkpoint-minutes", type=int, default=10, help="Periodic Drive backup interval (min)")
    parser.add_argument("--download-local", action="store_true", help="Also download final .tar.gz back to local ./outputs/")
    parser.add_argument("--local-out-dir", default=str(SCRIPT_DIR / "outputs"))
    parser.add_argument("--ainiee-ref", default="AiNiee7.3.0")
    parser.add_argument("--hf-token", default="", help="Optional Hugging Face token (or set HF_TOKEN env var) to avoid HF anonymous rate limits")
    parser.add_argument("--no-language-model-only", action="store_true", help="Disable vLLM --language-model-only (for non-multimodal custom models if needed)")
    parser.add_argument("--session", default="", help="Custom Colab session name")
    parser.add_argument("--keep-session", action="store_true", help="Debug only: do not stop Colab session on exit")
    args = parser.parse_args()

    input_path = Path(args.input_path).expanduser().resolve()
    input_files = collect_input_files(input_path)
    game_name = sanitize_name(args.game or (input_path.parent.name if input_path.is_file() else input_path.name))
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_name = f"{game_name}_{timestamp}"
    session_name = args.session or f"ainiee-{int(time.time()) % 100000}"

    gdrive_mode, rclone_conf_path = resolve_gdrive_mode(args.gdrive_mode, Path(args.rclone_conf).expanduser())
    want_local_download = args.download_local or gdrive_mode == "none"

    log(f"任务名称: {run_name}")
    log(f"输入文件: {[p.name for p in input_files]}")
    log(f"GPU: {args.gpu} | 模型档位: {args.model_preset} | Google Drive 模式: {gdrive_mode}")

    # Show current CU balance before starting
    usage_before = run_colab(["usage"], check=False, capture=True)
    if usage_before.stdout:
        for line in usage_before.stdout.strip().splitlines():
            log(f"Colab {line}")

    session_created = False
    try:
        # 1. Provision Colab GPU VM
        run_colab(["new", "-s", session_name, "--gpu", args.gpu])
        session_created = True

        # 2. Optional native Google Drive mount (interactive one-time click)
        if gdrive_mode == "drivemount":
            log("正在挂载 Google Drive（请在弹出的授权提示中确认一次，之后将全程无人值守）...")
            run_colab(["drivemount", "-s", session_name, "/content/drive"])

        # 3. Upload worker scripts, input MTool files, optional resume cache & rclone.conf
        log("上传执行脚本与 MTool 翻译文件到 Colab...")
        subprocess.run(
            ["colab", "exec", "-s", session_name],
            input="import os; os.makedirs('/content/work/input', exist_ok=True)\n",
            text=True,
            check=True,
        )
        run_colab(["upload", "-s", session_name, str(SCRIPT_DIR / "colab_worker.py"), "/content/colab_worker.py"])
        run_colab(["upload", "-s", session_name, str(SCRIPT_DIR / "ainiee_runner.py"), "/content/ainiee_runner.py"])

        for f in input_files:
            upload_file_auto_gz(session_name, f, f"/content/work/input/{f.name}")

        remote_resume = ""
        if args.resume_cache:
            resume_p = Path(args.resume_cache).expanduser().resolve()
            if not resume_p.is_file():
                raise FileNotFoundError(f"续翻缓存文件不存在: {resume_p}")
            remote_resume = "/content/work/resume_AinieeCacheData.json"
            upload_file_auto_gz(session_name, resume_p, remote_resume)

        if gdrive_mode == "rclone" and rclone_conf_path is not None:
            run_colab(["upload", "-s", session_name, str(rclone_conf_path), "/tmp/rclone.conf"])

        # 4. Build remote worker command
        # Note: if we need to colab download afterward, let local finally stop the VM instead of VM self-unassigning early.
        auto_unassign = (not args.keep_session) and (not want_local_download)
        worker_argv = [
            "python3",
            "/content/colab_worker.py",
            "--run-name",
            run_name,
            "--input-dir",
            "/content/work/input",
            "--work-dir",
            "/content/work",
            "--ainiee-ref",
            args.ainiee_ref,
            "--model-preset",
            args.model_preset,
            "--tokens-limit",
            str(args.tokens_limit),
            "--extract-token-limit",
            str(args.extract_token_limit),
            "--pre-line-counts",
            str(args.pre_line_counts),
            "--temperature",
            str(args.temperature),
            "--gdrive-mode",
            gdrive_mode,
            "--gdrive-dir",
            args.gdrive_dir,
            "--rclone-remote",
            args.rclone_remote,
            "--checkpoint-minutes",
            str(args.checkpoint_minutes),
        ]
        if args.model_id:
            worker_argv.extend(["--model-id", args.model_id])
        if args.threads > 0:
            worker_argv.extend(["--threads", str(args.threads)])
        if args.skip_extract:
            worker_argv.append("--skip-extract")
        if args.language_filter:
            worker_argv.append("--language-filter")
        if args.code_filter:
            worker_argv.append("--code-filter")
        if args.no_language_model_only:
            worker_argv.append("--no-language-model-only")
        if remote_resume:
            worker_argv.extend(["--resume-cache", remote_resume])

        hf_token = args.hf_token or os.environ.get("HF_TOKEN", "") or load_dotenv_value(SCRIPT_DIR / ".env", "HF_TOKEN")
        unassign_snippet = (
            "    try:\n"
            "        from google.colab import runtime as _rt\n"
            "        print('[Kernel] Calling google.colab.runtime.unassign()...', flush=True)\n"
            "        _rt.unassign()\n"
            "    except Exception as _e:\n"
            "        print(f'[Kernel] runtime.unassign() skipped: {_e}', flush=True)\n"
            if auto_unassign
            else "    pass\n"
        )
        exec_code = (
            "import os, subprocess, sys\n"
            f"argv = {json.dumps(worker_argv, ensure_ascii=False)}\n"
            "os.makedirs('/content/work', exist_ok=True)\n"
            "env = os.environ.copy()\n"
            "env['PYTHONUNBUFFERED'] = '1'\n"
            + (f"env['HF_TOKEN'] = {json.dumps(hf_token)}\n" if hf_token else "")
            + "rc = 1\n"
            "try:\n"
            "    with open('/content/work/worker.log', 'w', encoding='utf-8', buffering=1) as wlog:\n"
            "        proc = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, env=env)\n"
            "        assert proc.stdout is not None\n"
            "        for line in proc.stdout:\n"
            "            wlog.write(line)\n"
            "            wlog.flush()\n"
            "            print(line, end='', flush=True)\n"
            "        rc = proc.wait()\n"
            "finally:\n"
            + unassign_snippet
            + "if rc != 0:\n"
            "    raise RuntimeError(f'colab_worker failed with exit code {rc}')\n"
        )

        log("开始在 Colab 执行无人值守流水线（环境初始化 -> vLLM -> AiNiee 术语提取 -> 翻译 -> 导出保存）...")
        exec_proc = subprocess.Popen(
            ["colab", "exec", "-s", session_name, "--timeout", "0"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert exec_proc.stdin is not None and exec_proc.stdout is not None
        exec_proc.stdin.write(exec_code)
        exec_proc.stdin.close()

        seen_worker_start = False
        seen_worker_finish = False
        already_printed_bytes = 0
        for line in exec_proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            if "[Worker " in line:
                seen_worker_start = True
            if seen_worker_start and not seen_worker_finish:
                if line.startswith(("╭─", "│ ", "╰─", "Traceback ", "RuntimeError:")):
                    pass
                else:
                    already_printed_bytes += len(line.encode("utf-8"))
            if "FINAL SUMMARY: status=" in line:
                seen_worker_finish = True

        exec_rc = exec_proc.wait()
        monitor_outcome = ""
        if not seen_worker_finish:
            st_check = run_colab(["status", "-s", session_name], check=False, capture=True)
            if st_check.returncode == 0:
                monitor_outcome = monitor_detached_colab_session(session_name, already_printed_bytes, poll_sec=60)

        # 5. Optional local backup download
        report_status = None
        if want_local_download and monitor_outcome != "unassigned":
            local_out_dir = Path(args.local_out_dir).expanduser().resolve()
            local_out_dir.mkdir(parents=True, exist_ok=True)
            remote_archive = f"/content/work/deliverable/{run_name}.tar.gz"
            local_archive = local_out_dir / f"{run_name}.tar.gz"
            log(f"拉取结果压缩包到本地: {local_archive}")
            dl = run_colab(["download", "-s", session_name, remote_archive, str(local_archive)], check=False)
            if dl.returncode == 0 and local_archive.is_file():
                extract_dir = local_out_dir / run_name
                shutil.unpack_archive(str(local_archive), str(local_out_dir))
                log(f"已解压到本地目录: {extract_dir}")
                rep_file = extract_dir / "run_report.json"
                if rep_file.is_file():
                    report_status = json.loads(rep_file.read_text(encoding="utf-8")).get("status")
            else:
                log("WARNING: 本地拉取压缩包失败（如果已开启 Google Drive 模式，请直接检查云盘）。")

        if report_status and report_status != "succeeded":
            log(f"远程任务未成功完成: status={report_status}")
            return 1
        if monitor_outcome == "FAILED":
            log("远程任务以 FAILED 结束，请查看上方日志与 Google Drive 中的归档/断点。")
            return 1
        if monitor_outcome in ("SUCCEEDED", "unassigned"):
            log("全部流程完成！")
            return 0
        if exec_rc != 0 and not auto_unassign:
            log(f"远程任务返回非零退出码: {exec_rc}")
            return exec_rc

        log("全部流程完成！")
        return 0

    finally:
        if session_created and not args.keep_session:
            log(f"确保关闭 Colab 会话 '{session_name}'（防止继续扣费）...")
            run_colab(["stop", "-s", session_name], check=False)
            usage_after = run_colab(["usage"], check=False, capture=True)
            if usage_after.stdout:
                for line in usage_after.stdout.strip().splitlines():
                    log(f"Colab {line}")


if __name__ == "__main__":
    sys.exit(main())
