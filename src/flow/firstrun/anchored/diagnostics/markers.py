"""Which distribution each cutoff actually lives in, and what the pipeline expects of it.

Getting the *pool* right is the difference between a meaningful diagnostic and a
meaningless one. ``gate_with`` applies each cutoff inside a specific parent population; a
CD19 cutoff evaluated against all live events instead of the CD45+CD14- lymphocyte pool is
measuring a distribution the pipeline never looks at.

The ``accepted_fraction`` bands are NOT invented here — they are read off the pipeline's own
robust cut functions in ``gates.py``, which already encode what fraction-positive is
plausible per marker (and reject a candidate valley that falls outside). Reusing them means
the diagnostics import the pipeline's stated priors rather than smuggling in new ones.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

# Pool identifiers resolved against a gate_with() mask dict (plus two derived pools built in
# context.py, marked below).
POOL_SINGLETS = "sing"
POOL_LIVE = "live"
POOL_CD45P = "cd45p"
POOL_CD14N = "cd14n"
POOL_CD19N = "cd19n"
POOL_CD19N_CD3NEG = "cd19n_cd3neg"  # derived: cd19n & (cd3 < cd3_cut)
POOL_T = "T"
POOL_NK = "NK"
POOL_DONOR = "Donor"


@dataclass(frozen=True)
class MarkerSpec:
    """One cutoff the anchored pipeline sets, and everything needed to audit it.

    Attributes:
        key: cut key as used in the anchor's ``reference_cuts`` / the ``C`` dict.
        label: display label, matching ``unified_cutoffs.csv`` so the two tables join.
        pool: parent population the cutoff is DERIVED in (see module docstring).
        applied_pool: population the cutoff is APPLIED to in ``gate_with``. Differs from
            ``pool`` only for CD56, which is derived inside the CD3- subset but applied
            across the whole CD19- pool.
        positive_above: True when "positive" means above the cutoff. False for a dim-donor
            HLA cutoff, where donor cells are the DIM side.
        positive_label: what the above-cutoff side actually is, so a reader is never left
            guessing (for viability the above-cutoff population is DEAD, not live).
        accepted_fraction: the pipeline's own plausible range for the positive fraction, or
            None where gates.py states no band.
        valley_derived: whether the pipeline derives this cutoff from a valley in ``pool``.
            False for HLA/CAR, which come from controls or a longitudinal fit — their
            valley-shape metrics are informational, not a pass/fail criterion.
        downstream: reported metrics this cutoff can move, used to scope the sensitivity
            sweep to the populations that actually depend on it.
    """

    key: str
    label: str
    pool: str
    applied_pool: str
    positive_above: bool
    positive_label: str
    accepted_fraction: Optional[tuple[float, float]]
    valley_derived: bool
    downstream: tuple[str, ...]


# Headline longitudinal metrics — the numbers the study reports.
NK_METRIC = "%NK (of lymph)"
T_METRIC = "%T (of lymph)"
B_METRIC = "%B (of lymph)"
DONOR_METRIC = "%Donor NK (of lymph)"
DONOR_OF_NK_METRIC = "%Donor NK (of NK)"
CAR_METRIC = "%CAR+ (of Donor NK)"
CD4_METRIC = "%CD4 (of lymph)"
CD8_METRIC = "%CD8 (of lymph)"
LYMPH_METRIC = "%Lymphocytes (of CD45+)"
LIVE_METRIC = "%Live (of singlets)"
CD45_METRIC = "%CD45+ (of live)"
MONO_METRIC = "%CD14+ (of CD45+)"
PURITY_METRIC = "lineage_purity"

# Everything downstream of a cleanup cutoff moves when that cutoff moves.
_ALL_LINEAGE = (B_METRIC, T_METRIC, NK_METRIC, DONOR_METRIC, CAR_METRIC, PURITY_METRIC)

MARKERS: tuple[MarkerSpec, ...] = (
    MarkerSpec(
        key="ld",
        label="Viability(L/D)",
        pool=POOL_SINGLETS,
        applied_pool=POOL_SINGLETS,
        positive_above=True,
        positive_label="dead (bright L/D)",
        accepted_fraction=(0.01, 0.50),  # gates.robust_ld_cut
        valley_derived=True,
        downstream=(LIVE_METRIC, CD45_METRIC) + _ALL_LINEAGE,
    ),
    MarkerSpec(
        key="cd45",
        label="CD45",
        pool=POOL_LIVE,
        applied_pool=POOL_LIVE,
        positive_above=True,
        positive_label="CD45+ (leukocyte)",
        accepted_fraction=(0.02, 0.90),  # gates.robust_cd45_cut
        valley_derived=True,
        downstream=(CD45_METRIC, LYMPH_METRIC) + _ALL_LINEAGE,
    ),
    MarkerSpec(
        key="cd14",
        label="CD14",
        pool=POOL_CD45P,
        applied_pool=POOL_CD45P,
        positive_above=True,
        positive_label="CD14+ (monocyte)",
        accepted_fraction=(0.01, 0.40),  # gates.robust_cd14_cut max_frac
        valley_derived=True,
        downstream=(MONO_METRIC, LYMPH_METRIC) + _ALL_LINEAGE,
    ),
    MarkerSpec(
        key="cd19",
        label="CD19",
        pool=POOL_CD14N,
        applied_pool=POOL_CD14N,
        positive_above=True,
        positive_label="CD19+ (B cell)",
        accepted_fraction=(0.001, 0.12),  # gates.robust_cd19_cut
        valley_derived=True,
        downstream=(B_METRIC, T_METRIC, NK_METRIC, DONOR_METRIC, CAR_METRIC, PURITY_METRIC),
    ),
    MarkerSpec(
        key="cd3",
        label="CD3",
        pool=POOL_CD19N,
        applied_pool=POOL_CD19N,
        positive_above=True,
        positive_label="CD3+ (T cell)",
        accepted_fraction=(0.15, 0.90),  # gates.robust_cd3_cut
        valley_derived=True,
        downstream=(T_METRIC, NK_METRIC, DONOR_METRIC, CAR_METRIC, CD4_METRIC, CD8_METRIC,
                    PURITY_METRIC),
    ),
    MarkerSpec(
        key="cd56",
        label="CD56",
        pool=POOL_CD19N_CD3NEG,   # derived inside the CD3- pool (gates.robust_cd56_cut)
        applied_pool=POOL_CD19N,  # applied across CD19- (NK = CD19- CD3- CD56+)
        positive_above=True,
        positive_label="CD56+ (NK)",
        accepted_fraction=(0.04, 0.95),  # gates.robust_cd56_cut
        valley_derived=True,
        downstream=(NK_METRIC, T_METRIC, DONOR_METRIC, CAR_METRIC, PURITY_METRIC),
    ),
    MarkerSpec(
        key="cd4",
        label="CD4",
        pool=POOL_T,
        applied_pool=POOL_T,
        positive_above=True,
        positive_label="CD4+ (of T)",
        accepted_fraction=None,
        valley_derived=True,
        downstream=(CD4_METRIC, CD8_METRIC),
    ),
    MarkerSpec(
        key="hla",
        label="HLA (Donor)",
        pool=POOL_NK,
        applied_pool=POOL_NK,
        positive_above=True,  # overridden per-run when the anchor sets hla_dim
        positive_label="donor-HLA side",
        accepted_fraction=None,
        valley_derived=False,  # control-bracketed or fitted to manual, not a pool valley
        downstream=(DONOR_METRIC, DONOR_OF_NK_METRIC, CAR_METRIC),
    ),
    MarkerSpec(
        key="car",
        label="CAR",
        pool=POOL_DONOR,
        applied_pool=POOL_DONOR,
        positive_above=True,
        positive_label="CAR+",
        accepted_fraction=None,
        valley_derived=False,  # NT-NK p99.9 / product GMM / adaptive, not a pool valley
        downstream=(CAR_METRIC,),
    ),
)

MARKERS_BY_KEY = {m.key: m for m in MARKERS}

# CD8 is defined as CD4-negative within T, not by a cutoff of its own
# (anchor sets cd8=None, cd8_from_cd4_neg=True). Running placement metrics on it would be
# auditing a cutoff that does not exist, so it is excluded explicitly rather than by omission.
NO_CUTOFF_KEYS = frozenset({"cd8"})

# Metrics carried through the sensitivity + uncertainty tiers. Kept deliberately short: every
# entry multiplies the perturbation sweep, and these are the numbers the study reports.
HEADLINE_METRICS: tuple[str, ...] = (
    NK_METRIC,
    T_METRIC,
    B_METRIC,
    DONOR_METRIC,
    DONOR_OF_NK_METRIC,
    CAR_METRIC,
)

# Parent population (and its mask key) behind each headline metric, for counting statistics.
METRIC_PARENTS: dict[str, str] = {
    NK_METRIC: "lympho",
    T_METRIC: "lympho",
    B_METRIC: "lympho",
    CD4_METRIC: "lympho",
    CD8_METRIC: "lympho",
    DONOR_METRIC: "lympho",
    DONOR_OF_NK_METRIC: "NK",
    CAR_METRIC: "Donor",
    LYMPH_METRIC: "cd45p",
    CD45_METRIC: "live",
    LIVE_METRIC: "sing",
    MONO_METRIC: "cd45p",
}

# Numerator mask behind each headline metric.
METRIC_NUMERATORS: dict[str, str] = {
    NK_METRIC: "NK",
    T_METRIC: "T",
    B_METRIC: "B",
    CD4_METRIC: "CD4",
    CD8_METRIC: "CD8",
    DONOR_METRIC: "Donor",
    DONOR_OF_NK_METRIC: "Donor",
    CAR_METRIC: "CAR",
    LYMPH_METRIC: "lympho",
    CD45_METRIC: "cd45p",
    LIVE_METRIC: "live",
    MONO_METRIC: "CD14p",
}


def resolve_polarity(spec: MarkerSpec, hla_dim: bool) -> MarkerSpec:
    """Return ``spec`` with HLA polarity applied.

    With a dim-donor mismatch the anchor sets ``hla_dim`` and ``gate_with`` selects donor
    cells BELOW the HLA cutoff. Every positivity, placement and control metric has to follow
    that inversion or the donor checks read exactly backwards.
    """
    if spec.key != "hla" or not hla_dim:
        return spec
    return MarkerSpec(
        key=spec.key,
        label=spec.label,
        pool=spec.pool,
        applied_pool=spec.applied_pool,
        positive_above=False,
        positive_label="donor-HLA side (DIM — below the cutoff)",
        accepted_fraction=spec.accepted_fraction,
        valley_derived=spec.valley_derived,
        downstream=spec.downstream,
    )
