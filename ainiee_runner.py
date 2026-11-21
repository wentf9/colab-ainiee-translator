#!/usr/bin/env python3
"""Headless AiNiee 7.3+ runner executed inside the Colab AiNiee virtualenv.

Drives official NEKOparapa/AiNiee Backend.application.runtime.Runtime directly:
1. Configures local OpenAI-compatible vLLM endpoint (127.0.0.1:8000/v1).
2. Loads MTool JSON directory (or resumes an existing AinieeCacheData.json).
3. Runs 2-stage terminology / character / non-translate extraction.
4. Runs multi-threaded batch translation with extracted project tables injected.
5. Saves official AiNiee project cache and exports translated MTool files.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any


TERMINAL_STATES = {"succeeded", "partial", "failed", "cancelled", "interrupted"}

# Faithful, uncensored Galgame / RPG JP->ZH system prompt aligned with AiNiee's <textarea> parser.
GALGAME_SYSTEM_PROMPT_ZH = """你是一名顶级的日本 Galgame、轻小说与日式 RPG 本地化翻译专家。你的任务是将{source_language}文本逐行翻译为自然地道的{target_language}，不要合并或遗漏任何一行，并严格保留原文中的序号、换行符、转义符与游戏控制符（如 \\N[1]、\\V[1]、\\C[1]、%1、\\n 等）。

### 翻译原则
1. 忠实还原：原作是虚构的成人向艺术作品，允许并要求直接、准确、生动地翻译所有情感、粗口、亲密及 18x/R18 成人描写，严禁删减、回避、淡化、委婉化或道德说教。
2. 二次元语感：准确把握角色性格、口癖、敬语与亲疏关系；联系上下文正确使用人称代词，不擅自添加原文中不存在的主语；拟声词与喘息/感叹词按中文轻小说与 Galgame 习惯自然译出。
3. 术语统一：严格遵守提供的角色表、术语表与禁翻表，保持全篇译名一致。

### 以textarea标签输出译文
<textarea>
1.{target_language}文本
</textarea>"""


def log(msg: str) -> None:
    ts = time.strftime("%H:%M:%S")
    print(f"[AiNiee {ts}] {msg}", flush=True)


def wait_for_job(runtime: Any, job_id: str, phase: str, poll_interval: float = 3.0) -> dict[str, Any]:
    start = time.time()
    last_log = 0.0
    while True:
        job = runtime.scheduler.get(job_id)
        state = job.get("state", "unknown")
        now = time.time()
        if now - last_log >= 15.0 or state in TERMINAL_STATES:
            status = runtime.status()
            progress = status.get("progress") or {}
            line = progress.get("line", 0)
            total_line = progress.get("total_line", 0)
            tokens = progress.get("token", 0)
            elapsed = now - start
            extra = ""
            if total_line:
                pct = (line / total_line) * 100.0
                speed = line / elapsed if elapsed > 0 else 0.0
                extra = f" | {line}/{total_line} ({pct:.1f}%) | {speed:.2f} 行/秒 | tokens={tokens}"
            else:
                work_status = status.get("work_status")
                job_prog = job.get("progress") if isinstance(job.get("progress"), dict) else {}
                stage_msg = job_prog.get("stage") or job_prog.get("message") or ""
                if work_status:
                    extra += f" | work_status={work_status}"
                if stage_msg:
                    extra += f" | {stage_msg}"
            log(f"[{phase}] state={state} | elapsed={elapsed:.1f}s{extra}")
            last_log = now
        if state in TERMINAL_STATES:
            return job
        time.sleep(poll_interval)


def export_analysis_json(analysis_data: dict[str, Any] | None, out_path: Path) -> dict[str, int]:
    data = analysis_data if isinstance(analysis_data, dict) else {}
    cleaned = {
        "characters": data.get("characters", []) or [],
        "terms": data.get("terms", []) or [],
        "non_translate": data.get("non_translate", []) or [],
    }
    out_path.write_text(json.dumps(cleaned, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "characters": len(cleaned["characters"]),
        "terms": len(cleaned["terms"]),
        "non_translate": len(cleaned["non_translate"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run AiNiee 7.3+ headlessly against local vLLM.")
    parser.add_argument("--ainiee-dir", required=True, help="Path to cloned NEKOparapa/AiNiee repo")
    parser.add_argument("--input-dir", required=True, help="Directory containing MTool JSON file(s)")
    parser.add_argument("--output-dir", required=True, help="Directory for exported translated MTool files")
    parser.add_argument("--result-json", required=True, help="Path to write structured summary JSON")
    parser.add_argument("--glossary-json", required=True, help="Path to write extracted glossary JSON")
    parser.add_argument("--api-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--model-name", default="translator")
    parser.add_argument("--project-type", default="Mtool")
    parser.add_argument("--source-lang", default="japanese")
    parser.add_argument("--target-lang", default="chinese_simplified")
    parser.add_argument("--threads", type=int, default=24)
    parser.add_argument("--tokens-limit", type=int, default=1024)
    parser.add_argument("--extract-token-limit", type=int, default=8000)
    parser.add_argument("--pre-line-counts", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.35)
    parser.add_argument("--request-timeout", type=int, default=300)
    parser.add_argument("--round-limit", type=int, default=8)
    parser.add_argument("--skip-extract", action="store_true")
    parser.add_argument("--language-filter", action="store_true", help="Enable AiNiee language_filter_switch")
    parser.add_argument("--code-filter", action="store_true", help="Enable AiNiee code_filter_switch (Magika)")
    parser.add_argument("--resume-cache", default="", help="Optional existing AinieeCacheData.json to resume")
    parser.add_argument("--project-meta-file", default="", help="Optional file written as soon as project_id is known")
    args = parser.parse_args()

    ainiee_dir = Path(args.ainiee_dir).resolve()
    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Ensure AiNiee runs in portable headless mode anchored at ainiee_dir
    # Override Colab kernel's inherited MPLBACKEND (matplotlib_inline) with headless Agg
    os.environ["MPLBACKEND"] = "Agg"
    os.environ["AINIEE_APP_ROOT"] = str(ainiee_dir)
    os.environ["AINIEE_PORTABLE"] = "1"
    sys.path.insert(0, str(ainiee_dir))

    from Backend.application.runtime import Runtime
    from Backend.core.paths import config_path, project_cache_root
    from Backend.domain.prompts.prompt_builder_enum import PromptBuilderEnum

    log(f"Starting AiNiee headless Runtime from {ainiee_dir}")
    runtime = Runtime()
    runtime.start()

    platform_tag = "custom_platform_colab_vllm"
    summary: dict[str, Any] = {
        "status": "running",
        "project_id": "",
        "project_cache_dir": "",
        "config_file": str(config_path()),
        "output_dir": str(output_dir),
        "extracted_counts": {"characters": 0, "terms": 0, "non_translate": 0},
        "analysis_state": "skipped" if args.skip_extract else "pending",
        "translation_state": "pending",
        "export_state": "pending",
        "errors": [],
    }

    try:
        current_cfg = runtime.load_config()
        platforms = dict(current_cfg.get("platforms") or {})
        platforms[platform_tag] = {
            "tag": platform_tag,
            "group": "custom",
            "name": "Colab-vLLM",
            "api_url": args.api_url,
            "api_key": args.api_key,
            "api_format": "OpenAI",
            "icon": "LocalLLM",
            "user_thread_counts": args.threads,
            "rpm_limit": 60000,
            "model": args.model_name,
            "temperature": args.temperature,
            "think_switch": False,
            "think_depth": "low",
            "auto_complete": False,
            "model_datas": [args.model_name],
            "key_in_settings": [
                "api_url",
                "api_key",
                "model",
                "user_thread_counts",
                "rpm_limit",
                "temperature",
                "think_switch",
            ],
        }

        cfg_patch = {
            "platforms": platforms,
            "api_settings": {
                "active": platform_tag,
                "extract": platform_tag,
                "translate": platform_tag,
                "polish": platform_tag,
                "proofread": platform_tag,
            },
            "translation_project": args.project_type,
            "label_input_path": str(input_dir),
            "label_output_path": str(output_dir),
            "auto_set_output_path": False,
            "output_filename_suffix": "",
            "source_language": args.source_lang,
            "target_language": args.target_lang,
            "language_filter_switch": bool(args.language_filter),
            "code_filter_switch": bool(args.code_filter),
            "tokens_limit_switch": True,
            "lines_limit_switch": False,
            "tokens_limit": args.tokens_limit,
            "pre_line_counts": args.pre_line_counts,
            "request_timeout": args.request_timeout,
            "round_limit": args.round_limit,
            "extract_task_token_limit": args.extract_token_limit,
            "auto_extract_non_translate_switch": True,
            "extract_short_name_merge_switch": True,
            "few_shot_and_example_switch": True,
            "response_check_switch": {
                "return_to_original_text_check": True,
                "residual_original_text_check": True,
                "newline_character_count_check": True,
            },
            "translation_prompt_selection": {
                "last_selected_id": str(PromptBuilderEnum.COMMON),
                "prompt_content": GALGAME_SYSTEM_PROMPT_ZH,
            },
        }
        runtime.save_config(cfg_patch)
        log(f"Configured AiNiee platform={platform_tag} model={args.model_name} threads={args.threads}")

        # Quick API connectivity sanity check via AiNiee itself
        api_test_res = runtime.start_api_test(
            {
                "tag": platform_tag,
                "name": "Colab-vLLM",
                "api_url": args.api_url,
                "api_key": args.api_key,
                "api_format": "OpenAI",
                "model": args.model_name,
                "auto_complete": False,
            }
        )
        api_test_job = wait_for_job(runtime, api_test_res["job_id"], "api_test", poll_interval=1.0)
        log(f"AiNiee API test finished: state={api_test_job.get('state')}")
        if api_test_job.get("state") == "failed":
            raise RuntimeError(f"AiNiee API test failed: {api_test_job.get('error')}")

        # Load fresh MTool project or resume from existing AinieeCacheData.json
        continue_mode = False
        if args.resume_cache and Path(args.resume_cache).is_file():
            log(f"Resuming existing AiNiee cache: {args.resume_cache}")
            runtime.cache_call("load_from_file", [str(Path(args.resume_cache).resolve())], {})
            continue_mode = True
        else:
            log(f"Loading project type={args.project_type} from {input_dir}")
            runtime.load_project(args.project_type, str(input_dir), "")

        project_id = runtime.snapshot_id()
        if not project_id:
            raise RuntimeError("Failed to obtain AiNiee project_id after loading project")

        project_cache_dir = Path(project_cache_root()) / project_id
        summary["project_id"] = project_id
        summary["project_cache_dir"] = str(project_cache_dir)
        log(f"Loaded project_id={project_id} | cache_dir={project_cache_dir}")

        if args.project_meta_file:
            Path(args.project_meta_file).write_text(
                json.dumps(
                    {
                        "project_id": project_id,
                        "project_cache_dir": str(project_cache_dir),
                        "config_file": str(config_path()),
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

        # Stage 1: Terminology / character / non-translate extraction
        if not args.skip_extract:
            existing_analysis = runtime.cache.get_analysis_data() if runtime.cache else None
            has_existing = bool(
                isinstance(existing_analysis, dict)
                and (
                    existing_analysis.get("characters")
                    or existing_analysis.get("terms")
                    or existing_analysis.get("non_translate")
                )
            )
            if continue_mode and has_existing:
                log("Resumed project already has analysis_data; reusing existing terminology table.")
                summary["analysis_state"] = "reused"
                summary["extracted_counts"] = export_analysis_json(existing_analysis, Path(args.glossary_json))
            else:
                log("Stage 1/3: Running AiNiee terminology & character extraction...")
                t0 = time.time()
                ana_res = runtime.start_analysis()
                ana_job = wait_for_job(runtime, ana_res["job_id"], "extract", poll_interval=2.0)
                summary["analysis_state"] = ana_job.get("state", "unknown")
                summary["analysis_seconds"] = round(time.time() - t0, 2)
                if ana_job.get("error"):
                    summary["errors"].append({"stage": "extract", "error": ana_job.get("error")})
                runtime.save_project()
                analysis_data = runtime.cache.get_analysis_data()
                counts = export_analysis_json(analysis_data, Path(args.glossary_json))
                summary["extracted_counts"] = counts
                log(
                    f"Extraction complete ({summary['analysis_state']}): "
                    f"characters={counts['characters']}, terms={counts['terms']}, "
                    f"non_translate={counts['non_translate']}"
                )
        else:
            export_analysis_json({}, Path(args.glossary_json))

        # Stage 2: Batch translation
        log(f"Stage 2/3: Running AiNiee batch translation (continue={continue_mode})...")
        t1 = time.time()
        # Only pass continue_status=True if stats_data actually exists in the resumed cache
        use_continue = bool(continue_mode and getattr(runtime.cache.project, "stats_data", None) is not None)
        tr_res = runtime.start_translation(continue_status=use_continue)
        tr_job = wait_for_job(runtime, tr_res["job_id"], "translate", poll_interval=3.0)
        summary["translation_state"] = tr_job.get("state", "unknown")
        summary["translation_seconds"] = round(time.time() - t1, 2)
        if tr_job.get("error"):
            summary["errors"].append({"stage": "translate", "error": tr_job.get("error")})

        # Stage 3: Save official AiNiee project & export translated MTool files
        log("Stage 3/3: Saving official AiNiee project & exporting translated MTool files...")
        runtime.save_project()
        exp_res = runtime.start_export({"export_path": str(output_dir)})
        exp_job = wait_for_job(runtime, exp_res["job_id"], "export", poll_interval=1.0)
        summary["export_state"] = exp_job.get("state", "unknown")
        if exp_job.get("error"):
            summary["errors"].append({"stage": "export", "error": exp_job.get("error")})

        # Collect final stats
        final_status = runtime.status()
        summary["progress"] = final_status.get("progress")
        summary["exported_files"] = sorted(
            str(p.relative_to(output_dir)) for p in output_dir.rglob("*") if p.is_file()
        )
        summary["cache_files"] = sorted(
            str(p.relative_to(project_cache_dir))
            for p in project_cache_dir.rglob("*")
            if p.is_file()
        ) if project_cache_dir.exists() else []

        ok_states = {"succeeded", "partial"}
        summary["status"] = (
            "succeeded"
            if summary["translation_state"] in ok_states and summary["export_state"] in ok_states
            else "failed"
        )
        log(f"AiNiee pipeline finished: status={summary['status']} | exported={summary['exported_files']}")
        return 0 if summary["status"] == "succeeded" else 1

    except Exception as exc:
        summary["status"] = "failed"
        summary["errors"].append({"stage": "fatal", "error": repr(exc)})
        log(f"FATAL error in ainiee_runner: {exc!r}")
        raise
    finally:
        try:
            if runtime.snapshot_id():
                runtime.save_project()
        except Exception:
            pass
        try:
            runtime.close()
        except Exception:
            pass
        Path(args.result_json).write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


if __name__ == "__main__":
    sys.exit(main())
