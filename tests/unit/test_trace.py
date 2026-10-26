"""Component tests for TraceContext — the trace recording primitive.

Red/green contract (docs/spec-p10-trace.md, ticket 01):
- TraceContext(trace_type) records stage durations automatically (wall clock
  between record_stage calls); the caller never times anything manually
- finish() finalizes the trace; to_dict() returns a pure JSON-serializable dict
- stage order and the elapsed_ms per stage follow the caller's record order

This is the component under the application-service seam: the wiring tests
cover the full path through rag_service.query / IngestPipeline.ingest; these
tests pin the primitive's own contract (auto timing, finish, serializability)
which the wiring tests only touch indirectly.
"""

from __future__ import annotations

import json

from rag_mcp_server.core.trace import TraceContext


def test_trace_type_carried_and_fields_present():
    trace = TraceContext("query")
    trace.record_stage("query_processing", {})
    trace.finish()

    data = trace.to_dict()
    assert data["trace_type"] == "query"
    assert data["trace_id"]
    assert data["started_at"]
    assert data["finished_at"]
    assert isinstance(data["total_elapsed_ms"], (int, float))
    assert isinstance(data["stages"], list)
    assert data["stages"][0]["stage"] == "query_processing"


def test_record_stage_measures_elapsed_between_calls():
    trace = TraceContext("query")
    trace.record_stage("query_processing", {})
    trace.record_stage("dense_retrieval", {"hits": 3})
    trace.finish()

    data = trace.to_dict()
    assert data["stages"][0]["stage"] == "query_processing"
    assert data["stages"][1]["stage"] == "dense_retrieval"
    # each stage carries its own elapsed_ms (auto-measured)
    for stage in data["stages"]:
        assert "elapsed_ms" in stage
        assert isinstance(stage["elapsed_ms"], (int, float))


def test_record_stage_merges_data_payload():
    trace = TraceContext("query")
    trace.record_stage("dense_retrieval", {"hits": 3, "strategy": "dense"})
    trace.finish()

    stage = trace.to_dict()["stages"][0]
    assert stage["stage"] == "dense_retrieval"
    assert stage["data"]["hits"] == 3
    assert stage["data"]["strategy"] == "dense"


def test_explicit_elapsed_overrides_auto_measurement():
    trace = TraceContext("query")
    trace.record_stage("query_processing", {})
    trace.record_stage("fusion", {}, elapsed_ms=12.5)
    trace.finish()

    stages = trace.to_dict()["stages"]
    assert stages[1]["stage"] == "fusion"
    assert stages[1]["elapsed_ms"] == 12.5


def test_to_dict_is_json_serializable_and_round_trips():
    trace = TraceContext("ingestion")
    trace.record_stage("load", {"file": "a.pdf"})
    trace.record_stage("embed", {})
    trace.finish()

    encoded = json.dumps(trace.to_dict(), ensure_ascii=False)
    decoded = json.loads(encoded)
    assert decoded["trace_type"] == "ingestion"
    assert decoded["stages"][0]["data"]["file"] == "a.pdf"


def test_total_elapsed_covers_start_to_finish():
    trace = TraceContext("query")
    trace.record_stage("query_processing", {})
    trace.finish()

    data = trace.to_dict()
    assert data["total_elapsed_ms"] >= 0
    assert data["finished_at"] >= data["started_at"]


def test_ingestion_trace_type_accepted():
    trace = TraceContext("ingestion")
    trace.record_stage("load", {})
    trace.finish()
    assert trace.to_dict()["trace_type"] == "ingestion"


def test_metadata_written_when_provided():
    """C1: a trace constructed with metadata carries it top-level, so the
    query_traces page can label rows; update_metadata merges more fields."""
    trace = TraceContext("query", metadata={"query": "MCP"})
    trace.record_stage("query_processing", {})
    trace.update_metadata(result_count=2, top_k=3)
    trace.finish()

    data = trace.to_dict()
    assert data["metadata"] == {"query": "MCP", "result_count": 2, "top_k": 3}
    # JSON-serializable with the metadata field intact
    decoded = json.loads(json.dumps(data, ensure_ascii=False))
    assert decoded["metadata"]["query"] == "MCP"


def test_metadata_absent_when_none_provided():
    """C1: no metadata -> to_dict has no metadata key (ingestion trace shape
    unchanged; a Dict[str, Any] consumer that never set it sees the old shape)."""
    trace = TraceContext("ingestion")
    trace.record_stage("load", {})
    trace.finish()
    assert "metadata" not in trace.to_dict()
