#!/usr/bin/env bash
# 一次性配置本地 rclone（无需 sudo），用于让 Colab 无人值守自动上传 AiNiee 工程到 Google Drive。
set -euo pipefail

BIN_DIR="$HOME/.local/bin"
mkdir -p "$BIN_DIR"
export PATH="$BIN_DIR:$PATH"

if ! command -v rclone >/dev/null 2>&1; then
  OS_NAME="$(uname -s | tr '[:upper:]' '[:lower:]')"
  ARCH_RAW="$(uname -m)"
  case "$OS_NAME" in
    linux)  RCLONE_OS="linux" ;;
    darwin) RCLONE_OS="osx" ;;
    *)      echo "暂不支持的操作系统: $OS_NAME，请手动安装 rclone" >&2; exit 1 ;;
  esac
  case "$ARCH_RAW" in
    x86_64|amd64)  RCLONE_ARCH="amd64" ;;
    aarch64|arm64) RCLONE_ARCH="arm64" ;;
    *)             echo "暂不支持的 CPU 架构: $ARCH_RAW，请手动安装 rclone" >&2; exit 1 ;;
  esac
  echo "[1/2] 正在下载官方 rclone (${RCLONE_OS}-${RCLONE_ARCH}) 到 $BIN_DIR/rclone ..."
  TMP_DIR="$(mktemp -d)"
  trap 'rm -rf "$TMP_DIR"' EXIT
  curl -fsSL "https://downloads.rclone.org/rclone-current-${RCLONE_OS}-${RCLONE_ARCH}.zip" -o "$TMP_DIR/rclone.zip"
  python3 -c "import zipfile, sys; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])" "$TMP_DIR/rclone.zip" "$TMP_DIR"
  RCLONE_BIN="$(find "$TMP_DIR" -maxdepth 2 -type f -name rclone | head -n 1)"
  install -m 755 "$RCLONE_BIN" "$BIN_DIR/rclone"
  echo "rclone 安装完成: $($BIN_DIR/rclone version | head -n 1)"
else
  echo "[1/2] 已检测到 rclone: $(command -v rclone)"
fi

echo
echo "[2/2] 即将创建名为 'gdrive' 的 Google Drive 远程配置："
echo "      接下来会打开浏览器完成一次 Google 账号授权；授权完成后，"
echo "      今后所有 Colab 翻译任务都可完全零点击、无人值守自动上传到 Google Drive！"
echo

if rclone listremotes 2>/dev/null | grep -qx "gdrive:"; then
  echo "检测到已存在 'gdrive:' 配置，验证连通性..."
  rclone lsd gdrive: --max-depth 1 | head -n 10
  echo "配置正常！可直接运行 python3 run_colab_translate.py ..."
  exit 0
fi

rclone config create gdrive drive scope=drive

echo
echo "验证 Google Drive 连通性..."
rclone mkdir gdrive:AiNiee_Translations
rclone lsd gdrive:AiNiee_Translations
echo
echo "✅ 配置成功！配置文件位于: $HOME/.config/rclone/rclone.conf"
echo "   今后直接运行即可全自动保存工程到 Google Drive/AiNiee_Translations/ ："
echo "   python3 run_colab_translate.py /path/to/ManualTransFile.json --game \"游戏名\""
