#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR/src:${PYTHONPATH:-}"

if python -c "import mcp" 2>/dev/null; then
  INSTALL_CMD=(python -c "import mcp; print('mcp SDK 已安装: v2+')")
else
  echo "==> 未检测到 mcp SDK，尝试安装项目依赖"
  INSTALL_CMD=(python -m pip install -e .)
fi
VERIFY_CMD=(python scripts/verify.py)
START_CMD=(python -m rag_mcp_server)

echo "==> 当前目录: $PWD"
echo "==> Python 版本"
python --version

echo "==> 同步依赖"
"${INSTALL_CMD[@]}"

echo "==> 运行基础验证"
"${VERIFY_CMD[@]}"

echo "==> 启动命令"
printf '    %q' "${START_CMD[@]}"
printf '\n'

if [ "${RUN_START_COMMAND:-0}" = "1" ]; then
  echo "==> 启动应用"
  exec "${START_CMD[@]}"
fi

echo "如果希望 init.sh 直接启动应用，请设置 RUN_START_COMMAND=1。"