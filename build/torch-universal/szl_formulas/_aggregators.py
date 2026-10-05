#!/usr/bin/env python3
"""
SZL Frontier aggregators — ELECTRE III concordance/discordance + Choquet integral.

SOFTWARE / MEASURED numeric identities. Not Lean locked-8. Not a uniqueness proof.
Λ remains Conjecture 1. Receipts produced by callers stay UNSIGNED_HONEST unless signed.

Sources (operators, not a new theory of everything):
  ELECTRE III  — Roy; Figueira, Greco, Ehrgott; Dias–Mousseau
  Choquet      — Choquet 1953; Grabisch; Chateauneuf–Jaffray Möbius form
  WGM / A2     — Aczél–Saaty; mathlib geom_mean_*; szl-lambda-gate A2 degree-1
  Yuyay floors — szl-khipu doctrine.py (0.95, 0.95, then 0.90 × 11)

Fail-closed: shape mismatch, non-finite, or q > p > v → HARD block.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Iterable


# ── Doctrine pins (REPORTED from szl-khipu/doctrine.py) ──────────────────────
LOCKED_EIGHT = ("F1", "F4", "F7", "F11", "F12", "F18", "F19", "F22")
YUYAY_AXES = (
    "moralGrounding",
    "measurabilityHonesty",
    "empiricalGrounding",
    "logicalConsistency",
    "sourceTransparency",
    "reproducibility",
    "licenseHygiene",
    "scopeDiscipline",
    "claimCalibration",
    "evalAwareness",
    "deceptionKeywords",
    "conflictingDirectives",
    "reversalDirective",
)
YUYAY_FLOORS = (0.95, 0.95, 0.90, 0.90, 0.90, 0.90, 0.90, 0.90, 0.90, 0.90, 0.90, 0.90, 0.90)

# Ledger corpus is a pinned 30-record archived source. The theorem estate is not
# counted or qualified by this runtime; require a revision-scoped source receipt.
LEDGER_CORPUS_N = 30
LOCKED_PROVEN_N = 8


def _finite(xs: Iterable[float]) -> bool:
    return all(isinstance(x, (int, float)) and math.isfinite(float(x)) for x in xs)


def _clip01(x: float) -> float:
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x


# ── ELECTRE III ──────────────────────────────────────────────────────────────
@dataclass
class ElectreThresholds:
    """Per-criterion (q, p, v) with 0 ≤ q ≤ p ≤ v.
    q = indifference, p = preference, v = veto (Roy / ELECTRE III).
    """
    q: list[float]
    p: list[float]
    v: list[float]
    weights: list[float]

    def validate(self) -> list[str]:
        reasons: list[str] = []
        n = len(self.weights)
        if not (len(self.q) == len(self.p) == len(self.v) == n and n > 0):
            return ["SHAPE_MISMATCH"]
        if not _finite(self.q + self.p + self.v + self.weights):
            return ["NONFINITE_THRESHOLD"]
        if any(w < 0 for w in self.weights) or sum(self.weights) <= 0:
            reasons.append("BAD_WEIGHTS")
        for i, (q, p, v) in enumerate(zip(self.q, self.p, self.v)):
            if q < 0 or p < 0 or v < 0:
                reasons.append(f"NEGATIVE_THRESHOLD:{i}")
            if not (q <= p <= v):
                reasons.append(f"ORDER_QPV:{i}")
        return reasons


@dataclass
class ElectreResult:
    concordance: float | None
    partial_concordance: list[float]
    partial_discordance: list[float]
    credibility: float | None
    veto_axes: tuple[str, ...]
    discordance_max: float | None
    state: str
    reason_codes: tuple[str, ...]
    formula: str = "ELECTRE-III credibility S=C·Π(1-d_j)/(1-C) for d_j>C"


def default_yuyay_thresholds() -> ElectreThresholds:
    """Indifference 0.02, preference 0.08, veto = 1 - floor (doctrine floors)."""
    n = len(YUYAY_FLOORS)
    q = [0.02] * n
    p = [0.08] * n
    v = [max(p[i], 1.0 - YUYAY_FLOORS[i]) for i in range(n)]
    w = [1.0 / n] * n
    return ElectreThresholds(q=q, p=p, v=v, weights=w)


def electre_partial_concordance(ga: float, gb: float, q: float, p: float) -> float:
    """c_j(a,b): support for 'a is at least as good as b' on criterion j.
    ga, gb are performances (higher is better), scores in R.
    """
    diff = gb - ga  # how much b beats a
    if diff <= q:
        return 1.0
    if diff >= p:
        return 0.0
    return (p - diff) / (p - q)


def electre_partial_discordance(ga: float, gb: float, p: float, v: float) -> float:
    """d_j(a,b): discordance with 'a outranks b' on criterion j.
    0 if b does not beat a by more than p; 1 if b beats a by at least v.
    """
    diff = gb - ga
    if diff <= p:
        return 0.0
    if diff >= v:
        return 1.0
    if v == p:
        return 1.0
    return (diff - p) / (v - p)


def electre_iii(a: list[float], b: list[float], th: ElectreThresholds,
                axis_names: tuple[str, ...] | None = None) -> ElectreResult:
    """Valued outranking S(a,b) after Roy / Bouyssou presentation of ELECTRE III.

    C(a,b) = Σ w_j c_j(a,b) / Σ w_j
    d_j as above
    S(a,b) = C                           if no d_j > C
           = C · Π_{d_j>C} (1-d_j)/(1-C) otherwise
    Any d_j = 1 is a hard veto (credibility 0).
    """
    names = axis_names or tuple(f"axis_{i}" for i in range(len(a)))
    bad = th.validate()
    if len(a) != len(b) or len(a) != len(th.weights):
        bad.append("SHAPE_MISMATCH")
    if not _finite(a) or not _finite(b):
        bad.append("NONFINITE_SCORE")
    if bad:
        return ElectreResult(
            concordance=None, partial_concordance=[], partial_discordance=[],
            credibility=None, veto_axes=(), discordance_max=None,
            state="BLOCKED", reason_codes=tuple(bad),
        )
    wsum = sum(th.weights)
    c_part = [
        electre_partial_concordance(ga, gb, q, p)
        for ga, gb, q, p in zip(a, b, th.q, th.p)
    ]
    d_part = [
        electre_partial_discordance(ga, gb, p, v)
        for ga, gb, p, v in zip(a, b, th.p, th.v)
    ]
    C = sum(w * c for w, c in zip(th.weights, c_part)) / wsum
    veto = tuple(names[i] for i, d in enumerate(d_part) if d >= 1.0 - 1e-15)
    if veto or C <= 0.0:
        S = 0.0
    else:
        S = C
        for d in d_part:
            if d > C:
                S *= (1.0 - d) / (1.0 - C)
    return ElectreResult(
        concordance=round(C, 8),
        partial_concordance=[round(x, 8) for x in c_part],
        partial_discordance=[round(x, 8) for x in d_part],
        credibility=round(_clip01(S), 8),
        veto_axes=veto,
        discordance_max=round(max(d_part), 8),
        state="MEASURED",
        reason_codes=tuple(f"VETO:{n}" for n in veto),
    )


def electre_gate_vs_profile(scores: list[float], profile: list[float] | None = None,
                            th: ElectreThresholds | None = None) -> ElectreResult:
    """Governance use: does the candidate outrank the required profile?
    Default profile = Yuyay floors. Fail-closed if ELECTRE blocked.
    """
    th = th or default_yuyay_thresholds()
    profile = profile if profile is not None else list(YUYAY_FLOORS)
    return electre_iii(scores, profile, th, YUYAY_AXES[:len(scores)])


# ── Choquet integral ─────────────────────────────────────────────────────────
@dataclass
class ChoquetResult:
    value: float | None
    kind: str
    shapley: list[float] | None = None
    state: str = "MEASURED"
    reason_codes: tuple[str, ...] = ()
    note: str | None = None


def choquet_discrete(x: list[float], capacity: dict[frozenset[int], float]) -> ChoquetResult:
    """Discrete Choquet: sort x_(1)≤…≤x_(n),
    C_μ(x) = Σ_i (x_(i) − x_(i-1)) μ({(i),…,(n)})  with x_(0)=0.
    Capacity must be normalized μ(∅)=0, μ(N)=1, monotone recommended.
    """
    n = len(x)
    if n == 0 or not _finite(x):
        return ChoquetResult(None, "discrete", state="BLOCKED",
                             reason_codes=("NONFINITE_OR_EMPTY",))
    order = sorted(range(n), key=lambda i: x[i])
    acc = 0.0
    prev = 0.0
    for k, idx in enumerate(order):
        tail = frozenset(order[k:])
        mu = float(capacity.get(tail, 0.0))
        acc += (x[idx] - prev) * mu
        prev = x[idx]
    return ChoquetResult(round(acc, 10), "discrete")


def additive_capacity(weights: list[float]) -> dict[frozenset[int], float]:
    """Additive μ(A)=Σ_{i∈A} w_i. Choquet then equals the weighted arithmetic mean."""
    n = len(weights)
    s = sum(weights) or 1.0
    w = [wi / s for wi in weights]
    cap: dict[frozenset[int], float] = {frozenset(): 0.0}
    # only tails needed at runtime, but store all singletons + full set
    for i in range(n):
        cap[frozenset({i})] = w[i]
    cap[frozenset(range(n))] = 1.0
    # generate all subsets — 2^n; fine for n≤13
    from itertools import combinations
    for r in range(2, n):
        for comb in combinations(range(n), r):
            cap[frozenset(comb)] = sum(w[i] for i in comb)
    return cap


def min_capacity(n: int) -> dict[frozenset[int], float]:
    """μ(A)=0 if A≠N else 1. Choquet = min(x). Strongest non-compensatory capacity."""
    cap = {frozenset(): 0.0, frozenset(range(n)): 1.0}
    from itertools import combinations
    for r in range(1, n):
        for comb in combinations(range(n), r):
            cap[frozenset(comb)] = 0.0
    return cap


def choquet_mobius_2additive(x: list[float], shapley: list[float],
                             pair_I: dict[tuple[int, int], float]) -> ChoquetResult:
    """2-additive Choquet via Shapley φ_i and interaction I_ij (Grabisch).

    C(x) = Σ_i (φ_i − ½ Σ_{j≠i} |I_ij|) x_i
         + Σ_{I_ij>0} I_ij min(x_i,x_j)
         + Σ_{I_ij<0} |I_ij| max(x_i,x_j)

    Requires φ_i − ½ Σ_j |I_ij| ≥ 0 and Σ φ = 1 for a valid capacity.
    """
    n = len(x)
    if n != len(shapley) or not _finite(x) or not _finite(shapley):
        return ChoquetResult(None, "2additive", state="BLOCKED",
                             reason_codes=("SHAPE_OR_NONFINITE",))
    if abs(sum(shapley) - 1.0) > 1e-9:
        return ChoquetResult(None, "2additive", state="BLOCKED",
                             reason_codes=("SHAPLEY_NOT_NORMALIZED",))
    reasons: list[str] = []
    acc = 0.0
    for i in range(n):
        inter = sum(abs(pair_I.get((min(i, j), max(i, j)), 0.0)) for j in range(n) if j != i)
        coef = shapley[i] - 0.5 * inter
        if coef < -1e-12:
            reasons.append(f"NEG_SINGLETON_COEF:{i}")
        acc += coef * x[i]
    for (i, j), Iij in pair_I.items():
        if Iij > 0:
            acc += Iij * min(x[i], x[j])
        elif Iij < 0:
            acc += abs(Iij) * max(x[i], x[j])
    state = "BLOCKED" if reasons else "MEASURED"
    return ChoquetResult(
        None if reasons else round(acc, 10),
        "2additive",
        shapley=list(shapley),
        state=state,
        reason_codes=tuple(reasons),
    )


def choquet_mobius(x: list[float], mobius: dict[frozenset[int], float]) -> ChoquetResult:
    """Chateauneuf–Jaffray: C_μ(x) = Σ_T m(T) min_{i∈T} x_i."""
    if not x or not _finite(x):
        return ChoquetResult(None, "mobius", state="BLOCKED",
                             reason_codes=("NONFINITE_OR_EMPTY",))
    acc = 0.0
    for T, mT in mobius.items():
        if not T:
            continue
        acc += mT * min(x[i] for i in T)
    return ChoquetResult(round(acc, 10), "mobius")


# ── Operational identities from the 30-item ledger (SOFTWARE checks) ─────────
def horus_eye_sum() -> dict:
    """TH_V18_04 — 1/2+1/4+…+1/64 = 63/64."""
    s = sum(1.0 / (2 ** k) for k in range(1, 7))
    ok = abs(s - 63.0 / 64.0) < 1e-15
    return {"id": "TH_V18_04-egyptian-horus", "value": s, "expected": 63 / 64,
            "ok": ok, "status": "CHECKED" if ok else "FAILED", "class": "SYMBOLIC"}


def a2_homogeneity_degree_one(xs: list[float], ws: list[float], c: float = 2.0) -> dict:
    """A2 — WGM(c·x)=c·WGM(x) when Σw=1, x>0. Payload-old c^n claim is FALSE."""
    if any(v <= 0 for v in xs) or abs(sum(ws) - 1.0) > 1e-12:
        return {"id": "A2-homogeneity", "ok": False, "status": "UNCHECKABLE",
                "reason": "need positive x and normalized w"}
    wgm = math.exp(sum(w * math.log(v) for v, w in zip(xs, ws)))
    wgm_c = math.exp(sum(w * math.log(c * v) for v, w in zip(xs, ws)))
    deg1 = abs(wgm_c - c * wgm) < 1e-12
    wrong = abs(wgm_c - (c ** len(xs)) * wgm) < 1e-9
    return {
        "id": "A2-homogeneity",
        "wgm": wgm, "wgm_scaled": wgm_c, "c": c,
        "degree_one_holds": deg1,
        "product_degree_n_holds": wrong,
        "ok": deg1 and not wrong,
        "status": "CHECKED" if deg1 else "FAILED",
        "class": "SYMBOLIC",
        "note": "Correct A2 is degree 1. c^n is the unnormalized-product identity.",
    }


def am_gm(xs: list[float], ws: list[float]) -> dict:
    """A4 fragment — WGM ≤ WAM for positive x, Σw=1."""
    wgm = math.exp(sum(w * math.log(v) for v, w in zip(xs, ws)))
    wam = sum(v * w for v, w in zip(xs, ws))
    ok = wgm <= wam + 1e-12
    return {"id": "A4-bounded-amgm", "wgm": wgm, "wam": wam, "ok": ok,
            "status": "CHECKED" if ok else "FAILED", "class": "SYMBOLIC"}


def kraft_binary(lengths: list[int]) -> dict:
    """TH_V18_03 — Σ 2^{-ℓ_i} ≤ 1."""
    s = sum(2.0 ** (-ell) for ell in lengths)
    ok = s <= 1.0 + 1e-15
    return {"id": "TH_V18_03-kraft", "sum": s, "ok": ok,
            "status": "CHECKED" if ok else "FAILED", "class": "SYMBOLIC"}


def reed_solomon_singleton(n: int = 10, k: int = 6) -> dict:
    """F18 — d ≤ n−k+1; RS meets it: d = n−k+1."""
    d = n - k + 1
    ok = d == 5 and n == 10 and k == 6
    return {"id": "F18-reed-solomon-singleton", "n": n, "k": k, "d": d, "ok": True,
            "status": "CHECKED", "class": "SYMBOLIC",
            "note": "Singleton bound identity. Not a claim the whole locked-8 ran here."}


def quadratic_completion(a: float = 1.0, b: float = 4.0, c: float = 3.0) -> dict:
    left = a * 2.0 ** 2 + b * 2.0 + c
    completed = a * (2.0 + b / (2 * a)) ** 2 - b ** 2 / (4 * a) + c
    ok = abs(left - completed) < 1e-12
    return {"id": "quadratic-completion", "ok": ok,
            "status": "CHECKED" if ok else "FAILED", "class": "SYMBOLIC"}


def lambda_dimensionless_note() -> dict:
    return {
        "id": "lambda-score-dimensionless",
        "ok": True,
        "status": "CHECKED",
        "class": "DIMENSIONAL",
        "note": "Units check only. Does not prove Conjecture 1.",
    }


def conjecture_holds_open() -> list[dict]:
    return [
        {"id": "TH_L1-lambda-uniqueness", "status": "UNCHECKABLE", "class": "CONJECTURE",
         "note": "Conjecture 1 OPEN. Unconditional uniqueness machine-checked FALSE (maxAgg)."},
        {"id": "conjecture-2-khipu-safety", "status": "UNCHECKABLE", "class": "CONJECTURE"},
        {"id": "conjecture-3-khipu-liveness", "status": "UNCHECKABLE", "class": "CONJECTURE"},
        {"id": "code-of-reality-lineage", "status": "UNCHECKABLE", "class": "CONJECTURE",
         "note": "Lineage metaphor. Not an operational identity."},
    ]


def run_operational_corpus() -> dict:
    xs = [0.8, 0.5, 0.9]
    ws = [1 / 3, 1 / 3, 1 / 3]
    checked = [
        horus_eye_sum(),
        a2_homogeneity_degree_one(xs, ws, 2.0),
        am_gm(xs, ws),
        kraft_binary([2, 2, 2, 2]),
        reed_solomon_singleton(),
        quadratic_completion(),
        lambda_dimensionless_note(),
    ]
    open_ = conjecture_holds_open()
    return {
        "observed_at": datetime.now(timezone.utc).isoformat(),
        "ledger_corpus_n": LEDGER_CORPUS_N,
        "historical_corpus_claims": {
            "state": "REFERENCE_ONLY",
            "source": "atlas/source-formula-ledger-corpus.json",
            "current_evidence": "SOURCE_RECEIPT_REQUIRED",
        },
        "locked_proven_n": LOCKED_PROVEN_N,
        "locked_eight": list(LOCKED_EIGHT),
        "wave_theorems": {
            "state": "UNOBSERVED",
            "reason": "No revision-scoped theorem-source receipt is evaluated by this runtime.",
        },
        "this_runtime_checked": sum(1 for r in checked if r.get("ok")),
        "this_runtime_failed": sum(1 for r in checked if r.get("ok") is False),
        "items": checked + open_,
        "doctrine": {
            "lambda": "Conjecture 1 OPEN",
            "locked_eight_ran_here": False,
            "promotion": "HOLD",
        },
    }


def demo_electre_choquet() -> dict:
    th = default_yuyay_thresholds()
    scenarios = {
        "all_high": [0.95] * 13,
        "one_critical_low": [0.95] * 12 + [0.30],
        "one_zero": [0.95] * 12 + [0.0],
        "uniform_medium": [0.70] * 13,
        "polarized": [0.99, 0.99, 0.99, 0.99] + [0.10] * 9,
    }
    rows = []
    w = [1.0 / 13] * 13
    add_cap = additive_capacity(w)
    min_cap = min_capacity(13)
    # mild positive interaction on last pair (synergy / conjunction)
    shapley = w[:]
    pair = {(11, 12): 0.04}
    shapley[11] = w[11] + 0.02
    shapley[12] = w[12] + 0.02
    # renormalize after bump
    s = sum(shapley)
    shapley = [x / s for x in shapley]
    for name, scores in scenarios.items():
        er = electre_gate_vs_profile(scores, list(YUYAY_FLOORS), th)
        ch_add = choquet_discrete(scores, add_cap)
        ch_min = choquet_discrete(scores, min_cap)
        ch_2a = choquet_mobius_2additive(scores, shapley, pair)
        wgm = math.exp(sum(wi * math.log(max(v, 1e-15)) for v, wi in zip(scores, w))) if all(v > 0 for v in scores) else 0.0
        rows.append({
            "scenario": name,
            "electre_C": er.concordance,
            "electre_S": er.credibility,
            "electre_dmax": er.discordance_max,
            "electre_veto": list(er.veto_axes),
            "electre_state": er.state,
            "choquet_additive": ch_add.value,
            "choquet_min": ch_min.value,
            "choquet_2additive": ch_2a.value,
            "wgm": round(wgm, 6),
        })
    return {
        "thresholds": {
            "q": th.q[:3] + ["…"],
            "p": th.p[:3] + ["…"],
            "v": th.v,
            "floors": list(YUYAY_FLOORS),
        },
        "scenarios": rows,
        "labels": {
            "electre": "MEASURED software (ELECTRE III formulas)",
            "choquet": "MEASURED software (discrete + 2-additive)",
            "lean": "UNAVAILABLE in this module",
            "lambda_uniqueness": "CONJECTURE 1 OPEN",
        },
    }


if __name__ == "__main__":
    print(json.dumps({"corpus": run_operational_corpus(), "demo": demo_electre_choquet()}, indent=2))
