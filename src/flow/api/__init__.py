"""FastAPI backend for FLOW.

Exposes project/upload/validate/run/artifact endpoints for the web UI. The API NEVER
runs agent code on the host — it delegates every run to the Docker NotebookEnvironment
via ``flow.runner``. Uploads are validated, size-limited, and traversal-checked.
"""

# NB: only re-export the factory. Importing the module-level ``app`` instance here would
# shadow the ``flow.api.app`` submodule (name collision), so we deliberately don't.
from flow.api.app import create_app

__all__ = ["create_app"]
