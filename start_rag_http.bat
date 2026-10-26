@echo off
chcp 65001 >nul
rem ===== RAG knowledge base HTTP server (:8123) =====
rem 使用仓库自带的 .venv；复用 rag_service / DocumentManager。
rem 若需离线联调（不起 Ollama），可加环境变量：set RAG_HTTP_EMBEDDING_PROVIDER=fake
cd /d "%~dp0"
set PYTHONPATH=%CD%\src
".venv\Scripts\python.exe" -m rag_mcp_server.http_server --host 127.0.0.1 --port 8123
pause
