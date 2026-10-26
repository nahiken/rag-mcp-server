# RAG-MCP-SERVER

从空白目录开始实现模块化 RAG MCP Server 的学习型项目。已完成**阶段 1（最小 RAG 闭环）+ 阶段 9（真实 RAG 扩展）+ P10（可观测性 / 质量 / 收口）**：配置系统、真实 Embedding（本地 Ollama）、向量库（Chroma）、PDF 加载（含图片提取与语义 caption）、混合检索（dense + sparse + RRF）、LLM 重排、Transform 增强、文件级增量跳过、查询编排、MCP 工具、CLI 适配、文档生命周期、Trace 打点、评估门禁、Dashboard 观测、端到端验证。

## 快速启动（推荐）

无需任何外部服务，一条命令跑通「建 demo 数据 → 摄取 → 查询」：

```bash
# PowerShell / cmd / bash 通用（Python 版，自动处理路径与编码）
python scripts/demo.py

# 单独操作
python scripts/demo.py ingest          # 重建并摄取 demo 数据
python scripts/demo.py query 检索       # 查询（默认词：检索）
python scripts/demo.py query MCP --json # JSON 输出
```

> demo 数据在系统临时目录，不污染仓库。脚本自动隔离配置（`strategy: sparse` + `persist_dir` 在临时目录），用 `--embedding-provider fake` 离线运行，不连 Ollama。

（Git Bash 用户也可用 bash 版 `./scripts/demo.sh`。）

## CLI 用法

```bash
export PYTHONPATH="$PWD/src"        # Windows Git Bash；或每命令带 PYTHONPATH=src

# 导入文档（txt/md/pdf → 向量库 + chunks.jsonl）
python -m rag_mcp_server ingest <路径> --embedding-provider fake

# 查询（默认读配置 retrieval.strategy=hybrid；离线建议配 sparse）
python -m rag_mcp_server query "检索"
python -m rag_mcp_server query "检索" --json --top-k 5

# 脚本入口（等价）
python scripts/ingest.py <路径> --embedding-provider fake
python scripts/query.py "检索" --json
```

## MCP server

```bash
# 通过 stdio 暴露 4 个工具：
#   query_knowledge_hub   —— 配置驱动查询（返回 Markdown 富响应 + JSON 引用 + 图片块）
#   semantic_search       —— 强制 dense 语义检索
#   list_collections      —— 列出已摄取集合（集合名 + 文档/chunk 数）
#   get_document_summary  —— 单文档富摘要（source_path / chunk 数 / 关联图片）
python -m rag_mcp_server.mcp_server.server

# 用官方 SDK 客户端验证
python scripts/mcp_client.py "检索"
```

## 多集合（Multi-Collection）

- 每个集合一个独立 Chroma collection（ADR 0016）。ingest 用 `--collection` 导入到指定集合，query 用 `collection:` 语法或 `--collection` 参数限定集合（参数优先）。
- 过滤语法：查询里可写 `source:xxx` / `tag:xxx` / `doc_type:xxx`，按 metadata 字段在集合内过滤（service 层解析，CLI 与 MCP 通用）。
- 默认集合叫 `default`（settings `vector_store.collection`）。旧 `data/chroma` 里的 `rag_chunks` 与旧 `data/index` 是历史遗留，不迁移。

## 测试

```bash
python scripts/verify.py          # 标准验证入口：smoke + pytest unit/e2e（排除 integration 与 dashboard smoke）
PYTHONPATH=src python -m pytest   # 全量：577 collected + 14 deselected（含 dashboard smoke，需 streamlit dev extra）
PYTHONPATH=src python -m pytest tests/unit/test_sparse_retrieval.py -q   # 单个功能
PYTHONPATH=src python -m pytest -m integration tests/integration         # 真实 Ollama（需本地 127.0.0.1:11451）
```

> `verify.py` 用 `--ignore=tests/e2e/test_dashboard_smoke.py` 排除 streamlit 依赖，未装 dev extras 也能离线全绿（569 passed）；安装 streamlit 后全量 577 通过。Dashboard 冒烟在 `test_dashboard_smoke.py`（AppTest 密封），随全量 pytest 一起跑。

## Dashboard（观测）

```bash
# 启动本地观测台（streamlit，dev-only 可选依赖）
python scripts/dashboard.py        # 或 streamlit run src/rag_mcp_server/observability/dashboard/app.py
```

两页：**总览**（settings 配置卡片 + 知识库文档/chunk 数字卡）、**摄取追踪**（按文件聚合的 load/split/transform/embed/upsert 阶段耗时瀑布图 + 明细表，skip/integrity_skip 文件标记）。六页面扩展点在 ADR 0014。需要 `pip install "streamlit>=1.36"`（dev extras）。

## 评估（离线检索门禁）

```bash
python scripts/evaluate.py --test-set tests/fixtures/golden_test_set.json
# 默认门禁：hit_rate=1.0 / mrr=0.0（golden fixture 在 tests/fixtures/eval_docs/）
# 失败以非零退出码报告；Ragas 为可选 Answer 扩展（不启用，缺依赖清晰 ImportError）
```

## 多模态

- **PDF 摄取**：markitdown 文本路径 + pdfplumber 独立提图（ADR 0010），图片存 `data/images/` 并以 `[IMAGE: id]` 占位符进入嵌入文本。
- **图片 caption**：`ingestion.transform.image_captioner` 默认关闭，settings 设 `enabled: true` 时对 `[IMAGE: id]` 产出确定性 caption；`use_vision: true` 时经本地 Ollama Vision LLM 产出语义 caption（软降级，图片缺失/失败不阻断文本摄取）。
- **MCP 返回**：查询命中带图 chunk 时返回 `ImageContent` 图片块（`core/response/multimodal_assembler.py`）。

## 环境

- 依赖：`pytest`、`pyyaml`、`chromadb`、`markitdown[pdf]`、`jieba`、官方 `mcp` SDK；Dashboard 需 dev-only `streamlit>=1.36`。安装绕过坏代理：`python -m pip install --proxy "" <pkg>`。
- 真实 Embedding / 混合检索集成测试需要本地 Ollama（`nomic-embed-text` @ `127.0.0.1:11451`）。
- 项目文档：本 README 覆盖运行方式；内部设计记录（ADR、spec、进度日志）只保留在本地，不随仓库发布。
