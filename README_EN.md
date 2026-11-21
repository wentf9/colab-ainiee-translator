# Unattended Colab MTool + AiNiee 7.3 LLM Translation Pipeline

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB.svg)](https://www.python.org/)
[![Google Colab CLI](https://img.shields.io/badge/Google%20Colab-G4%20%7C%20A100-F9AB00.svg)](https://colab.research.google.com/)

**English** | [简体中文](README.md)

Run **vLLM + uncensored Qwen3.8/3.6 open-source LLMs + the official headless [AiNiee](https://github.com/NEKOparapa/AiNiee) 7.3.0 backend** completely unattended on Google Colab GPUs (default: **G4 — NVIDIA RTX PRO 6000 Blackwell 96GB**), providing a one-click pipeline for translating Japanese Galgames and RPGs (including 18x / R18 content) exported via MTool:

1. Automatically provisions a Colab GPU session and bootstraps isolated environments (`vLLM` venv in ~22.5s; `AiNiee` headless env installs in parallel during model weight download)
2. Launches a local vLLM server (native Blackwell FP8 + Prefix Caching + `--language-model-only` + server-side thinking disabled)
3. Loads MTool's `ManualTransFile.json` via the official AiNiee 7.3.0 runtime
4. Runs two-stage **AI terminology / character / non-translate control-code extraction** and injects the extracted tables into translation prompts
5. Executes high-concurrency batch translation (faithful uncensored 18x / R18 translation with RPG control codes preserved intact)
6. Backs up checkpoints to Google Drive every 10 minutes; upon completion, packages and uploads the **official AiNiee project + exported MTool translations** to Google Drive
7. Automatically shuts down the Colab runtime via both `google.colab.runtime.unassign()` and `colab stop` to prevent idle compute-unit charges

---

## Repository Structure

- `run_colab_translate.py`: Local one-click launcher (with transparent gzip upload for large files & disconnect polling recovery)
- `colab_worker.py`: Colab-side orchestrator (env bootstrapping, vLLM lifecycle, periodic Drive backup, packaging, auto-unassign)
- `ainiee_runner.py`: Headless runner driving the official `NEKOparapa/AiNiee` v7.3.0 backend
- `setup_gdrive.sh`: One-time local `rclone` Google Drive setup helper (no `sudo` required; supports Linux & macOS)
- `sample_mtool/ManualTransFile.json`: Synthetic MTool sample containing character names, fantasy terminology, RPG control codes, and Galgame dialogue
- `.env.example`: Optional `HF_TOKEN` environment variable template
- `README.md` / `README_EN.md`: Documentation in Simplified Chinese and English
- `LICENSE`: MIT License

---

## Prerequisites

- **OS**: Linux or macOS (per `google-colab-cli` platform support)
- **Python**: 3.10+
- **Google Colab Account & CLI**: A Google Colab account with GPU access and `google-colab-cli` installed:

```bash
# Option 1: Using uv (recommended)
uv tool install google-colab-cli

# Option 2: Using pip
pip install google-colab-cli

# Run any command once to complete Google Colab OAuth sign-in and verify compute-unit balance
colab usage
```

- **Optional: Configure a Hugging Face Token** (recommended to avoid anonymous download rate limits):

```bash
cp .env.example .env
# Edit .env and set HF_TOKEN=hf_xxx (or pass --hf-token / set HF_TOKEN env var)
```

---

## Quick Start

### 1. One-Time Google Drive Setup (Recommended for 100% Zero-Click Unattended Runs)

```bash
./setup_gdrive.sh
```

Follow the browser prompt once to authorize your Google account; this creates the `gdrive:` remote in `~/.config/rclone/rclone.conf`. All future jobs will upload automatically without interaction to `Google Drive/AiNiee_Translations/<game_name>_<timestamp>/` (temporary credentials are wiped from Colab immediately upon completion).

> Prefer not to configure `rclone`? Pass `--gdrive-mode drivemount` instead (uses native `colab drivemount`, requiring one browser confirmation at startup, then runs unattended).

### 2. Translate MTool Files with One Command

```bash
# Default: G4 (96GB, 10.05 CU/h) + Qwen3.8-27B-abliterated-FP8 (highest overall quality)
python3 run_colab_translate.py /path/to/ManualTransFile.json --game "MyGame"

# For large RPGs, enable --language-filter to skip pure ASCII / non-Japanese lines and save CU
python3 run_colab_translate.py /path/to/ManualTransFile.json --game "LargeRPG" --language-filter

# You can also pass a directory containing multiple MTool JSON files
python3 run_colab_translate.py /path/to/mtool_folder --game "MyGame"

# Maximum throughput / lowest cost per word: switch to MoE (Qwen3.6-35B-A3B-FP8, 3B active params)
python3 run_colab_translate.py /path/to/ManualTransFile.json --game "LargeRPG" --model-preset fast

# Also download a local backup archive to ./outputs/
python3 run_colab_translate.py /path/to/ManualTransFile.json --game "MyGame" --download-local

# Local download only, without uploading to Google Drive (great for smoke tests)
python3 run_colab_translate.py ./sample_mtool/ManualTransFile.json --game "SmokeTest" --gdrive-mode local
```

### 3. Resume from Checkpoint (If a Huge Job Is Interrupted)

Download `_checkpoint/AinieeCacheData.json` or `AiNiee_Project/AinieeCacheData.json` from Google Drive and pass `--resume-cache` to continue untranslated entries while automatically reusing the extracted glossary:

```bash
python3 run_colab_translate.py /path/to/ManualTransFile.json \
  --game "MyGame" \
  --resume-cache /path/to/AinieeCacheData.json
```

---

## Model Presets (`--model-preset`)

| Preset | Hugging Face Model | Size / VRAM | Default Concurrency | Best For |
| :--- | :--- | :--- | :---: | :--- |
| `quality` (default) | `munekazu/Huihui-Qwen3.8-27B-abliterated-FP8` | 28.8 GB (27.55 GiB on G4, ~55.06 GiB KV Cache) | 24 | **Best overall quality**: Galgame prose style, complex RPG plots, and accurate terminology extraction |
| `fast` | `coolthor/Huihui-Qwen3.6-35B-A3B-abliterated-FP8-DYNAMIC` | 35.8 GB MoE (only 3B active) | 32 | Ultra-fast batch translation for massive RPGs / Galgames with the lowest CU cost per word |
| `bf16` | `huihui-ai/Huihui-Qwen3.8-27B-abliterated` | 55.5 GB native BF16 | 16 | Zero quantization loss (recommended for G4 96GB only) |

> You can also specify any vLLM-compatible Hugging Face repository via `--model-id <org/repo>`.

---

## Common CLI Options

| Option | Default | Description |
| :--- | :--- | :--- |
| `--game` | Input folder name | Project / game name used for Drive folders and archive names |
| `--gpu` | `G4` | Colab GPU tier: `G4` / `A100` / `L4` / `T4` |
| `--model-preset` | `quality` | Model preset: `quality` / `fast` / `bf16` |
| `--model-id` | *(none)* | Custom Hugging Face model repo ID (overrides preset) |
| `--threads` | Preset default | AiNiee concurrent request threads |
| `--language-filter` | Off | Enable AiNiee `language_filter_switch` (skip non-source-language / pure ASCII entries) |
| `--code-filter` | Off | Enable AiNiee `code_filter_switch` (Magika-based code filtering) |
| `--skip-extract` | Off | Skip Stage 1 AI terminology & character extraction |
| `--resume-cache` | *(none)* | Path to an existing `AinieeCacheData.json` to resume translation |
| `--gdrive-mode` | `auto` | Google Drive mode: `auto` / `rclone` / `drivemount` / `local` |
| `--gdrive-dir` | `AiNiee_Translations` | Destination root directory in Google Drive |
| `--checkpoint-minutes` | `10` | Interval (in minutes) for periodic Drive checkpoint backups |
| `--download-local` | Off | Additionally download the final `.tar.gz` archive to `./outputs/` |
| `--hf-token` | `HF_TOKEN` / `.env` | Hugging Face Access Token |
| `--no-language-model-only` | Off | Disable vLLM `--language-model-only` flag |
| `--keep-session` | Off | Debug only: keep the Colab session alive on exit |

---

## Cold-Start Optimizations & Long-Job Reliability

### Cold-Start Optimizations (Tuned for Colab G4 96GB)

- **Skip unused vision encoder** (`--language-model-only`): Eliminates multimodal vision warmup and encoder cache profiling while saving ~0.9 GiB of weight VRAM for larger KV cache concurrency (disable with `--no-language-model-only`).
- **Skip first-run JIT & autotune overhead** (`--disable-flashinfer-autotune`, `VLLM_USE_FLASHINFER_SAMPLER=0`, `VLLM_USE_DEEP_GEMM=0`): Avoids cold-start FlashInfer sampler JIT compilation and full-bucket autotuning delays.
- **Overlapped environment setup & weight download**: Installs the `vLLM` virtualenv first to kick off model download immediately, while cloning `AiNiee` and installing its headless dependencies (~12.5s) runs concurrently in a background thread.
- **High-performance Hugging Face Xet downloads**: Enables `HF_XET_HIGH_PERFORMANCE=1` to saturate Colab datacenter bandwidth (especially when paired with `HF_TOKEN`).

### Reliability for Large Files & Multi-Hour Runs

- **Transparent gzip upload for large files**: Input `ManualTransFile.json` or resume caches larger than 2 MB are automatically gzip-compressed locally before upload and decompressed on Colab to avoid Colab's HTTP 413 payload limit.
- **Automatic WebSocket disconnect recovery**: Remote logs and state are streamed continuously to `/content/work/worker.log` and `/content/work/worker.status`. If `colab exec` drops mid-run during a multi-hour job, the local launcher automatically switches to a 60-second polling monitor that tails remote logs until completion and guarantees VM shutdown.
- **Periodic checkpoints & fault-tolerant archival**: Progress caches sync incrementally to `_checkpoint/` every 10 minutes. Even if an error occurs, partial progress is packaged and uploaded to Google Drive before dual-layer session termination.

---

## Deliverable Structure (Google Drive / Local Extract)

```text
<game_name>_<timestamp>/
├── output/
│   └── ManualTransFile.json          # Drop back into MTool to play immediately
├── AiNiee_Project/
│   ├── AinieeCacheData.json          # Official AiNiee 7.3 cache (open directly in desktop AiNiee to proofread/re-export)
│   ├── config.json                   # Task configuration snapshot
│   └── ProjectCache/<project_id>/    # Full official project cache folder (copy into desktop AiNiee/ProjectCache/)
├── extracted_glossary.json           # Human-readable extracted glossary (characters / terms / non_translate)
├── run_report.json                   # Timings, token counts, GPU info, and run status summary
├── vllm_tail.log                     # Tail of the vLLM inference server log
├── worker.log                        # Complete Colab worker execution log
└── <game_name>_<timestamp>.tar.gz    # Complete archive bundle
```

---

## Acknowledgements & Disclaimer

Built on top of these open-source projects and tools:

- [NEKOparapa/AiNiee](https://github.com/NEKOparapa/AiNiee) — Open-source AI game & text translation tool
- [vLLM](https://github.com/vllm-project/vllm) — High-throughput LLM inference and serving engine
- [google-colab-cli](https://pypi.org/project/google-colab-cli/) — Official Google Colab command-line interface
- [rclone](https://rclone.org/) — Cloud storage sync utility

> This project is intended for personal study and lawful game localization workflows. Please comply with Google Colab's Terms of Service, the licenses of any models you run, and the copyrights of the games you translate.

---

## License

Released under the [MIT License](LICENSE).
