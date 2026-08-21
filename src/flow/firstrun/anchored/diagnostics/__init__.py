"""Cutoff diagnostics for the anchored CAR-NK pipeline — "picture to numbers".

The anchored first-run derives one cutoff per marker at a reference timepoint and transfers it
to every other timepoint. The analysis specification then asks the agent to judge whether each
cutoff "is derived reasonably and holds across timepoints" by opening the overlay PNGs — a
visual judgement a text-only model cannot make, and will otherwise fabricate.

This package computes, from the event-level data, the quantities a cytometrist reads off those
figures, organized as six tiers that mirror the chain of trust:

    tier 0  data adequacy        counts, retention, Wilson intervals, acquisition stability
    tier 1  cutoff foundation    bimodality tests, valley depth, placement, derivation
                                 fingerprint — at the reference timepoint
    tier 2  transfer validity    negative-population drift, the technical-vs-biological
                                 discriminator, 2-D gate geometry — at every timepoint
    tier 3  sensitivity          local fragility, locked-vs-per-sample counterfactual, and a
                                 cutoff-attributable range on every headline number
    tier 4  internal controls    host NK (CAR-negative by construction), pre-infusion donor,
                                 NT-NK and CAR-product tubes
    tier 5  operator concordance per-timepoint / per-metric manual agreement and its coverage

Two rules hold everywhere. Quantities are computed from the same event arrays that generated
the figures — never from the rendered pixels — but in the figures' own coordinate system, so a
reported number is checkable by eye against the PNG. And the package MEASURES and FLAGS; it
never concludes. Each flag carries the rule that produced it and what would resolve it.

A third concern sits alongside those two: a report that counts everything honestly can still
be too large to finish. ``obligations.py`` derives a short, capped WORKLIST from the flags
that fired, so the output ends in a finite amount of work rather than a tally.

Nothing here modifies the first-run. Entry points::

    python -m anchored.diagnostics --data <DATA> --out <OUT> [--plots <PLOTS>]
    from anchored.diagnostics import run_diagnostics
"""

__all__ = ["run_diagnostics", "LIMITATIONS", "ANCHOR_FILENAME"]


def __getattr__(name):  # PEP 562 lazy access: importing this package must stay cheap
    if name in __all__:
        from .orchestrate import ANCHOR_FILENAME, LIMITATIONS, run_diagnostics

        return {
            "run_diagnostics": run_diagnostics,
            "LIMITATIONS": LIMITATIONS,
            "ANCHOR_FILENAME": ANCHOR_FILENAME,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
