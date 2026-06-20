"""FLOW — a faithful, minimal re-implementation of the Finch data-analysis agent.

FLOW is an autonomous, Jupyter-native data-analysis agent. Given a dataset and a
research question, it writes original, executable code line-by-line inside a Docker
container, observing outputs and adapting — exactly like Finch (FutureHouse). It is a
*generative reasoning engine*, not a template/retrieval system: there is **no**
hardcoded analysis logic anywhere in this package.

The codebase provides only:
  * an Aviary-shaped Environment (``flow.aviary`` + ``flow.env``),
  * exactly two agent tools (``edit_cell`` and ``submit_answer``),
  * a ReAct agent loop (``flow.agent``),
  * a Docker execution runtime,
  * a FastAPI backend (``flow.api``) and a React UI (``frontend/``).

See ``README.md`` for the Finch fidelity checklist and ``DESIGN.md`` for the ADRs.
"""

__version__ = "1.0.0"
