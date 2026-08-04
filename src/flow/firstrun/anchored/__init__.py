"""CAR-NK flow cytometry analysis pipeline (FSA × SSA first).
"""

__all__ = ["process_patient", "main"]


def __getattr__(name):  # PEP 562 lazy attribute access
    if name in ("process_patient", "main"):
        from .run import main, process_patient

        return {"process_patient": process_patient, "main": main}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
