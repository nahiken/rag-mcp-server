#!/usr/bin/env bash
# 一键启动 RAG-MCP-SERVER 离线 demo（无需 Ollama）。
# 用法：
#   ./scripts/demo.sh            # 完整跑一遍：建 demo 数据 -> ingest -> query
#   ./scripts/demo.sh ingest     # 只重建并摄取 demo 数据
#   ./scripts/demo.sh query 词    # 对已摄取数据查询（默认词: 检索）
#   ./scripts/demo.sh shell       # 进入预置好 PYTHONPATH 的子 shell
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# Python 是 Windows 原生程序，不认识 Git Bash 的 /d/... 路径；转成 D:/...
if command -v cygpath >/dev/null 2>&1; then
  ROOT_DIR="$(cygpath -m "$ROOT_DIR")"
fi
DEMO_DIR="${RAG_DEMO_DIR:-/tmp/rag-demo}"
if command -v cygpath >/dev/null 2>&1; then
  DEMO_DIR="$(cygpath -m "$DEMO_DIR")"
fi
# Windows Python 把 "D:/.../src:"（尾随冒号）当无效路径；只在有既有值时加冒号
export PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

demo_data() {
  mkdir -p "$DEMO_DIR/docs" "$DEMO_DIR/config"
  cat > "$DEMO_DIR/docs/notes.txt" <<'EOF'
RAG 混合检索演示，支持中文关键词检索。

MCP 是 Model Context Protocol，用于连接工具与数据。

语义检索用向量，关键词检索用 BM25。
EOF
  cat > "$DEMO_DIR/config/settings.yaml" <<EOF
vector_store:
  persist_dir: $DEMO_DIR/chroma
retrieval:
  strategy: sparse
  top_k: 3
EOF
  echo "==> demo 数据已写入 $DEMO_DIR"
}

demo_ingest() {
  demo_data
  (cd "$DEMO_DIR" && python -m rag_mcp_server ingest docs \
      --embedding-provider fake --index-dir out --json)
  echo "==> 摄取完成。查询示例："
  echo "    ./scripts/demo.sh query 检索"
  echo "    ./scripts/demo.sh query MCP --json"
}

demo_query() {
  local term="${1:-检索}"
  shift || true
  (cd "$DEMO_DIR" && python -m rag_mcp_server query "$term" --index-dir out "$@")
}

demo_shell() {
  echo "进入 demo 子 shell（已设 PYTHONPATH，cwd=$DEMO_DIR）。exit 退出。"
  cd "$DEMO_DIR"
  exec bash
}

cmd="${1:-run}"
case "$cmd" in
  run)      demo_ingest && echo "" && demo_query "检索" ;;
  ingest)   demo_ingest ;;
  query)    shift; demo_query "$@" ;;
  shell)    demo_shell ;;
  *)        echo "用法: $0 [run|ingest|query <词>|shell]" >&2; exit 1 ;;
esac
