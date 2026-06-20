"""Persistent in-container execution server (the "Jupyter kernel").

This module is intentionally **standalone** (no ``flow`` imports) so it can be copied
into the BixBench-env Docker image and run as ``python kernel_server.py`` with only the
standard library + scientific stack available inside the container.

Protocol (line-delimited JSON over stdin/stdout):
    host -> server : {"code": "<python source>"}\\n
    server -> host : <one JSON object>\\n<SENTINEL>\\n

Each request is executed in a **persistent global namespace**, giving true notebook
semantics: variables, imports, and loaded data survive across cells. Stdout/stderr are
captured; the last expression's repr is reported; matplotlib figures are saved to the
plots directory and summarized as ``[image: <path>]`` (raw image bytes are never returned).

The same execution function is reused on the host by ``flow.env.kernel.InProcessKernel``
in tests, so there is exactly one code path for "run a cell".
"""

from __future__ import annotations

import ast
import contextlib
import io
import json
import os
import sys
import traceback
from typing import Any

SENTINEL = "<<<FLOW_CELL_END>>>"


def _save_open_figures(plots_dir: str) -> list[str]:
    """Persist any open matplotlib figures to ``plots_dir``; return relative paths.

    Returns an empty list if matplotlib is not present or no figures are open. This is
    generic plumbing — it imposes *no* analysis semantics on what is plotted.
    """
    saved: list[str] = []
    if "matplotlib" not in sys.modules:
        return saved
    try:  # pragma: no cover - depends on container libs
        import matplotlib.pyplot as plt  # type: ignore

        nums = plt.get_fignums()
        if not nums:
            return saved
        os.makedirs(plots_dir, exist_ok=True)
        for num in nums:
            fig = plt.figure(num)
            existing = len(os.listdir(plots_dir))
            name = f"figure_{existing + 1:03d}.png"
            path = os.path.join(plots_dir, name)
            fig.savefig(path, bbox_inches="tight", dpi=110)
            saved.append(path)
        plt.close("all")
    except Exception:
        return saved
    return saved


def execute_code(namespace: dict[str, Any], code: str, plots_dir: str) -> dict[str, Any]:
    """Execute ``code`` in ``namespace``; return a structured, JSON-safe result.

    The body is executed as statements; if the final node is an expression, its repr is
    captured as ``result`` (notebook "last expression" behaviour). Errors are caught and
    returned as a traceback string rather than raised.
    """
    stdout, stderr = io.StringIO(), io.StringIO()
    result_repr = ""
    error = False
    tb = ""

    try:
        parsed = ast.parse(code, mode="exec")
    except SyntaxError:
        return {
            "stdout": "",
            "stderr": "",
            "result": "",
            "images": [],
            "error": True,
            "traceback": traceback.format_exc(),
        }

    last_expr = None
    if parsed.body and isinstance(parsed.body[-1], ast.Expr):
        last_expr = parsed.body.pop()

    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
        try:
            if parsed.body:
                exec(compile(parsed, "<cell>", "exec"), namespace)  # noqa: S102
            if last_expr is not None:
                expr_code = compile(
                    ast.Expression(last_expr.value), "<cell>", "eval"
                )
                value = eval(expr_code, namespace)  # noqa: S307
                if value is not None:
                    result_repr = repr(value)
        except Exception:
            error = True
            tb = traceback.format_exc()

    images = _save_open_figures(plots_dir)
    return {
        "stdout": stdout.getvalue(),
        "stderr": stderr.getvalue(),
        "result": result_repr,
        "images": images,
        "error": error,
        "traceback": tb,
    }


def main() -> None:  # pragma: no cover - exercised only inside the container
    """Run the request/response loop until stdin closes."""
    plots_dir = os.environ.get("FLOW_PLOTS_DIR", os.path.join(os.getcwd(), "plots"))
    namespace: dict[str, Any] = {"__name__": "__flow_cell__"}
    # Make the read-only dataset path discoverable without baking in any analysis.
    namespace["FLOW_DATA_DIR"] = os.environ.get("FLOW_DATA_DIR", "/data")
    namespace["FLOW_WORK_DIR"] = os.environ.get("FLOW_WORK_DIR", os.getcwd())

    sys.stdout.write("FLOW_KERNEL_READY\n")
    sys.stdout.flush()

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError:
            continue
        code = req.get("code", "")
        out = execute_code(namespace, code, plots_dir)
        sys.stdout.write(json.dumps(out) + "\n")
        sys.stdout.write(SENTINEL + "\n")
        sys.stdout.flush()


if __name__ == "__main__":  # pragma: no cover
    main()
