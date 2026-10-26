"""Observability dashboard package.

Streamlit is an optional/dev-only dependency: only ``app`` imports it at module
top level; ``pages`` render functions import it lazily inside their bodies. The
``pages`` and ``services`` modules are streamlit-free at top level (ADR 0014) so
the pure data/rendering logic stays testable by pytest and the six-page
extension point (data_browser / query_traces / evaluation_panel) does not depend
on the runtime.
"""
