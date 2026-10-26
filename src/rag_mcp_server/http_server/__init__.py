"""HTTP (REST) transport for the RAG MCP Server.

Composes the existing use-case layer (rag_service.ingest/query) and
DocumentManager into a Starlette app so mall-admin / mall-agent can ingest,
manage and query the knowledge base over HTTP without going through MCP stdio.
No business logic lives here — only request/response mapping.
"""
