"""Dashboard pages — render functions, streamlit-free at module top level.

ADR 0014: pages never import streamlit at module top level; each render
function imports it lazily inside the function body so the module stays
importable by plain pytest and the six-page extension point (data_browser /
query_traces / evaluation_panel) stays free of the streamlit runtime. Render
functions take injectable data args (``settings`` / ``trace_dir``) so any
frontend can drive the same services layer hermetically.
"""
