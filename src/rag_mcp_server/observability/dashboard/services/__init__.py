"""Dashboard services — pure data access, streamlit-free (ADR 0014).

These modules read persisted artifacts (traces.jsonl, chunks.jsonl, settings,
vector store stats) and return plain data. They never import streamlit, so
pytest can test them directly and any future frontend can render the same data.
"""
