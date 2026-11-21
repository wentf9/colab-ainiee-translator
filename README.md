# Colab 无人值守 MTool + AiNiee 7.3 开源大模型翻译流水线

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB.svg)](https://www.python.org/)
[![Google Colab CLI](https://img.shields.io/badge/Google%20Colab-G4%20%7C%20A100-F9AB00.svg)](https://colab.research.google.com/)

在 Google Colab **G4（NVIDIA RTX PRO 6000 Blackwell 96GB）** 等 GPU 运行时上全自动运行 **vLLM + 去审查 Qwen3.8/3.6 开源模型 + 官方 [AiNiee](https://github.com/NEKOparapa/AiNiee) 7.3.0 无头后端**，一键完成日语 Galgame / RPG 游戏（含 18x / R18 内容）的 MTool 翻译全流程：

1. 自动创建 Colab GPU 会话并快速配置隔离环境（vLLM 环境 ~22.5s；AiNiee 环境在后台与权重下载并行完成）
2. 本地拉起 vLLM（Blackwell 原生 FP8 + Prefix Caching + `--language-model-only` + 服务端关闭思考模式）
3. 官方 AiNiee 7.3.0 自动载入 MTool 导出的 `ManualTransFile.json`
4. 两阶段 **AI 术语 / 角色 / 禁翻控制符提取**，自动挂载到翻译提示词
5. 高并发批量翻译（含 18x / R18 原汁原味无审查翻译、RPG 控制符原样保留）
6. 每 10 分钟定时将断点缓存备份到 Google Drive；翻译完成后自动打包 **官方 AiNiee 工程 + MTool 译文** 上传至 Google Drive
7. 自动调用 `google.colab.runtime.unassign()` + `colab stop` 双保险关闭 Colab 会话，零空转扣费

---

## 目录说明

- `run_colab_translate.py`：本地一键启动入口（含大文件透明压缩上传与断线轮询恢复）
- `colab_worker.py`：Colab 端总控脚本（环境引导、vLLM、定时云盘备份、打包上传、自动停机）
- `ainiee_runner.py`：驱动官方 `NEKOparapa/AiNiee` v7.3.0 后端的无头执行器
- `setup_gdrive.sh`：一次性配置 `rclone` Google Drive 授权脚本（无需 sudo，支持 Linux / macOS）
- `sample_mtool/ManualTransFile.json`：含角色名、奇幻术语、RPG 控制符与 Galgame 对话的合成测试样本
- `.env.example`：可选的 `HF_TOKEN` 环境变量示例
- `LICENSE`：MIT 开源协议

---

## 环境准备

- **操作系统**：Linux 或 macOS（受 `google-colab-cli` 平台支持限制）
- **Python**：3.10+
- **Colab 账号与 CLI**：拥有可用 GPU 算力的 Google Colab 账号，并安装官方 `google-colab-cli`：

```bash
# 方式一：使用 uv（推荐）
uv tool install google-colab-cli

# 方式二：使用 pip
pip install google-colab-cli

# 首次运行任意命令完成 Google Colab OAuth 授权并确认算力余额
colab usage
```

- **可选：配置 Hugging Face Token**（推荐，避免 HF 匿名下载并发限速）：

```bash
cp .env.example .env
# 编辑 .env，填入 HF_TOKEN=hf_xxx（或在命令行通过 --hf-token / 环境变量传入）
```

---

## 快速开始

### 1. 一次性配置 Google Drive（推荐，之后永久零点击无人值守）

```bash
./setup_gdrive.sh
```

按提示在浏览器完成一次 Google 账号授权后，会在 `~/.config/rclone/rclone.conf` 生成 `gdrive:` 配置。今后每次翻译都会自动无交互上传到 `Google Drive/AiNiee_Translations/<游戏名>_<时间戳>/`（临时凭证用完自动从 Colab 删除）。

> 不想配 `rclone` 也可以直接加 `--gdrive-mode drivemount`（启动时通过官方 `colab drivemount` 点一次浏览器授权，后面全程无人值守）。

### 2. 一键无人值守翻译 MTool 文件

```bash
# 默认：G4 (96GB, 10.05 CU/h) + Qwen3.8-27B-abliterated-FP8（最高综合质量）
python3 run_colab_translate.py /path/to/ManualTransFile.json --game "你的游戏名"

# 大型 RPG 推荐开启 --language-filter，自动跳过纯英文/数字/非日语条目，大幅提速省算力
python3 run_colab_translate.py /path/to/ManualTransFile.json --game "大型RPG" --language-filter

# 如果是一个包含多个 MTool JSON 的目录，直接传目录路径即可
python3 run_colab_translate.py /path/to/mtool_folder --game "你的游戏名"

# 追求极致吞吐/省算力：切换 MoE 模型（Qwen3.6-35B-A3B-FP8，仅激活 3B 参数）
python3 run_colab_translate.py /path/to/ManualTransFile.json --game "大型RPG" --model-preset fast

# 同时也在本地 ./outputs/ 留一份备份
python3 run_colab_translate.py /path/to/ManualTransFile.json --game "你的游戏名" --download-local

# 仅下载到本地、不传云盘（适合测试）
python3 run_colab_translate.py ./sample_mtool/ManualTransFile.json --game "SmokeTest" --gdrive-mode local
```

### 3. 断点续翻（若超大工程意外中断）

从云盘的 `_checkpoint/AinieeCacheData.json` 或 `AiNiee_Project/AinieeCacheData.json` 下载到本地，传入 `--resume-cache` 即可接着未翻译条目继续翻（且自动复用已提取的术语表）：

```bash
python3 run_colab_translate.py /path/to/ManualTransFile.json \
  --game "你的游戏名" \
  --resume-cache /path/to/AinieeCacheData.json
```

---

## 模型档位（`--model-preset`）

| 档位 | Hugging Face 模型 | 体积 / 显存 | 并发默认 | 适用场景 |
| :--- | :--- | :--- | :---: | :--- |
| `quality`（默认） | `munekazu/Huihui-Qwen3.8-27B-abliterated-FP8` | 28.8 GB（G4 占 27.55 GiB，留 ~55.06 GiB KV Cache） | 24 | **综合质量天花板**，Galgame 文风、复杂 RPG 剧情、术语提取全能 |
| `fast` | `coolthor/Huihui-Qwen3.6-35B-A3B-abliterated-FP8-DYNAMIC` | 35.8 GB MoE（仅激活 3B） | 32 | 超长篇 RPG / Galgame 极速批量扫翻，单位字数 CU 成本最低 |
| `bf16` | `huihui-ai/Huihui-Qwen3.8-27B-abliterated` | 55.5 GB 原生 BF16 | 16 | 零量化无损精度（仅建议 G4 96GB） |

> 也支持通过 `--model-id <org/repo>` 指定任意兼容 vLLM 的 Hugging Face 模型。

---

## 常用参数说明

| 参数 | 默认值 | 说明 |
| :--- | :--- | :--- |
| `--game` | 输入目录名 | 项目 / 游戏名，用于云盘目录与归档命名 |
| `--gpu` | `G4` | Colab GPU 类型：`G4` / `A100` / `L4` / `T4` |
| `--model-preset` | `quality` | 预设模型档位：`quality` / `fast` / `bf16` |
| `--model-id` | 空 | 自定义 Hugging Face 模型 ID（覆盖预设） |
| `--threads` | 档位默认 | AiNiee 并发请求数 |
| `--language-filter` | 关闭 | 开启 AiNiee 语言过滤（跳过非源语言 / 纯 ASCII 条目） |
| `--code-filter` | 关闭 | 开启 AiNiee 代码过滤（基于 Magika） |
| `--skip-extract` | 关闭 | 跳过第一阶段 AI 术语/角色提取，直接开始翻译 |
| `--resume-cache` | 空 | 传入已有的 `AinieeCacheData.json` 进行断点续翻 |
| `--gdrive-mode` | `auto` | 云盘模式：`auto` / `rclone` / `drivemount` / `local` |
| `--gdrive-dir` | `AiNiee_Translations` | Google Drive 目标根目录 |
| `--checkpoint-minutes` | `10` | 定时备份断点缓存到云盘的间隔（分钟） |
| `--download-local` | 关闭 | 完成后额外将结果压缩包下载到本地 `./outputs/` |
| `--hf-token` | 读取 `HF_TOKEN`/`.env` | Hugging Face Token |
| `--no-language-model-only` | 关闭 | 关闭 vLLM `--language-model-only` 参数 |
| `--keep-session` | 关闭 | 仅调试用：结束时不自动关闭 Colab 会话 |

---

## 冷启动优化与长任务可靠性

### 冷启动优化（针对 Colab G4 96GB）

- **跳过未使用的视觉编码器**（`--language-model-only`）：省去多模态视觉预热与 Encoder Cache 探测，减少约 0.9 GiB 权重显存占用、为 KV Cache 腾出更大并发空间（可用 `--no-language-model-only` 关闭）。
- **跳过首次 JIT 编译与 Autotune**（`--disable-flashinfer-autotune`、`VLLM_USE_FLASHINFER_SAMPLER=0`、`VLLM_USE_DEEP_GEMM=0`）：避免启动时编译 FlashInfer Sampling JIT 与全桶 Autotune 耗时。
- **环境安装与模型下载并行**：先装好 vLLM 虚拟环境并立即拉取模型启动，AiNiee 无头环境克隆与依赖安装（~12.5s）完全在后台线程与模型下载并行完成。
- **启用 Hugging Face Xet 高速下载**：内置 `HF_XET_HIGH_PERFORMANCE=1`，配合 `HF_TOKEN` 跑满机房带宽。

### 大文件与无人值守可靠性

- **大文件透明压缩上传**：大于 2 MB 的 `ManualTransFile.json` 或续翻缓存会自动本地 gzip 压缩上传、远端解压，规避 Colab 上传接口的 HTTP 413 限制。
- **WebSocket 断线自动接管**：远程日志实时写入 `/content/work/worker.log` 与 `/content/work/worker.status`；即使数小时长任务期间本地 `colab exec` 连接意外断开，启动器也会自动切入每 60 秒一次的轮询守护模式，持续同步远程进度直至任务完成并确认释放会话。
- **定时断点 + 异常兜底归档**：默认每 10 分钟向云盘 `_checkpoint/` 增量同步缓存；即便发生异常中断，也会将已有进度打包上传云盘并触发双保险停机，防止白烧算力。

---

## 产出物结构（Google Drive / 本地解压后）

```text
<游戏名>_<时间戳>/
├── output/
│   └── ManualTransFile.json          # 直接放回 MTool 挂载即可游玩
├── AiNiee_Project/
│   ├── AinieeCacheData.json          # 官方 AiNiee 7.3 工程缓存（可直接拖入桌面版 AiNiee 打开校对/重导出）
│   ├── config.json                   # 本次任务配置快照
│   └── ProjectCache/<project_id>/    # 完整官方项目缓存目录（可直接复制进桌面版 AiNiee/ProjectCache/）
├── extracted_glossary.json           # 独立可读的提取结果（characters / terms / non_translate）
├── run_report.json                   # 耗时、token、GPU、状态汇总
├── vllm_tail.log                     # 推理后端日志摘要
├── worker.log                        # Colab 端完整运行日志
└── <游戏名>_<时间戳>.tar.gz          # 完整打包归档
```

---

## 致谢与声明

本项目基于以下优秀开源项目与工具构建：

- [NEKOparapa/AiNiee](https://github.com/NEKOparapa/AiNiee) — 开源游戏与文本 AI 翻译工具
- [vLLM](https://github.com/vllm-project/vllm) — 高吞吐大语言模型推理引擎
- [google-colab-cli](https://pypi.org/project/google-colab-cli/) — Google Colab 官方命令行工具
- [rclone](https://rclone.org/) — 云存储同步工具

> 本项目仅作为个人学习与合法本地化辅助工具提供，请遵守 Google Colab 服务条款、所选开源模型的对应许可协议以及游戏原作版权规定。

---

## License

本项目采用 [MIT License](LICENSE) 开源。
