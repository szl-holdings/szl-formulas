# SPDX-License-Identifier: Apache-2.0
"""EML real-domain research witness, independently implemented for SZL.

Paper: Andrzej Odrzywolek, All elementary functions from a single operator,
arXiv:2603.21852v2 (2026), CC BY 4.0, Eq. (3), (5), and Sec. 4.3.
Authoritative research implementation: github.com/VA00/SymbolicRegressionPackage.

Only three published shallow identities are tested here. This module is NOT
the author's compiler, a complete calculator, a symbolic identity proof, or
a gradient-trained model. It cannot be installed as a new locked SZL formula.
No GPU, model training, network, filesystem or native-code execution.
"""

from __future__ import annotations

import hashlib
import json
import re
from decimal import (
    Decimal,
    DecimalException,
    InvalidOperation,
    localcontext,
)

SCHEMA = "szl.formulas.eml-real-witness/v1"
PAPER_ID = "arxiv:2603.21852v2"
SOURCE_URL = "https://github.com/VA00/SymbolicRegressionPackage"
OPS = frozenset({"exp", "ln", "e"})
MAX_TREE_NODES = 31
MAX_DEPTH = 8
MAX_SAMPLES = 32
MAX_MAGNITUDE = Decimal("1e10")
MAX_EXPONENT_ARGUMENT = Decimal("50")
_TOLERANCE = Decimal("1e-24")
_DECIMAL_RE = re.compile(r"-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]{1,3})?\Z")


class EMLDomainError(ValueError):
    """Only a finite machine-readable refusal code, never raw sample values."""

    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


# The paper's primitive EML(x,y) = exp(x) - ln(y), with constant 1.
# These are the exact constructive forms of exp, ln and e stated in the paper.
_ONE = ("one",)
_X = ("var", "x")


def compile_known(operation: str) -> tuple:
    """Return a pure, immutable EML tree for one explicitly known identity."""
    if type(operation) is not str or operation not in OPS:
        raise EMLDomainError("OPERATION_UNSUPPORTED")
    if operation == "exp":
        return ("eml", _X, _ONE)
    if operation == "e":
        return ("eml", _ONE, _ONE)
    # ln(x) = EML(1, EML(EML(1, x), 1)); requires x > 0.
    return ("eml", _ONE, ("eml", ("eml", _ONE, _X), _ONE))


def _structure(tree: tuple) -> dict:
    """Validate exact-tree grammar with a depth/node cap before evaluation."""
    active: set[int] = set()
    seen = [0]

    def walk(node, depth: int):
        if depth > MAX_DEPTH:
            raise EMLDomainError("TREE_DEPTH_LIMIT")
        if not isinstance(node, tuple) or not node:
            raise EMLDomainError("TREE_SCHEMA_INVALID")
        if id(node) in active:
            raise EMLDomainError("TREE_CYCLE")
        seen[0] += 1
        if seen[0] > MAX_TREE_NODES:
            raise EMLDomainError("TREE_NODE_LIMIT")
        active.add(id(node))
        try:
            if node == _ONE:
                return {"op": "one"}, ["1"], depth
            if node == _X:
                return {"op": "var", "name": "x"}, ["x"], depth
            if len(node) != 3 or node[0] != "eml":
                raise EMLDomainError("TREE_SCHEMA_INVALID")
            l, lrpn, ld = walk(node[1], depth + 1)
            r, rrpn, rd = walk(node[2], depth + 1)
            return {"op": "eml", "left": l, "right": r}, lrpn + rrpn + ["E"], max(ld, rd)
        finally:
            active.remove(id(node))

    document, rpn, depth = walk(tree, 0)
    data = json.dumps(document, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return {
        "tree": document,
        "rpn": "".join(rpn),
        "nodes": seen[0],
        "operator_nodes": rpn.count("E"),
        "max_depth": depth,
        "tree_sha3_256": hashlib.sha3_256(data.encode("utf-8")).hexdigest(),
    }


def _decimal_input(value) -> Decimal:
    """Reject floats, NaNs, infinities, non-Decimal syntax and large input."""
    if type(value) is not str or not 1 <= len(value) <= 40 or _DECIMAL_RE.fullmatch(value) is None:
        raise EMLDomainError("INPUT_NOT_CANONICAL_DECIMAL")
    try:
        x = Decimal(value)
    except (InvalidOperation, ValueError):
        raise EMLDomainError("INPUT_NOT_CANONICAL_DECIMAL") from None
    if not x.is_finite() or abs(x) > 100:
        raise EMLDomainError("INPUT_OUT_OF_RANGE")
    return x


def _safe_eml(left: Decimal, right: Decimal) -> Decimal:
    """Bounded, *real-only* EML; no complex branch or extended-real coercion."""
    if not left.is_finite() or not right.is_finite():
        raise EMLDomainError("NONFINITE_NODE")
    if right <= 0:
        raise EMLDomainError("REAL_LOG_DOMAIN_REFUSED")
    if abs(left) > MAX_EXPONENT_ARGUMENT:
        raise EMLDomainError("EXPONENT_ARGUMENT_LIMIT")
    try:
        result = left.exp() - right.ln()
    except DecimalException:
        raise EMLDomainError("DECIMAL_OPERATION_REFUSED") from None
    if not result.is_finite() or abs(result) > MAX_MAGNITUDE:
        raise EMLDomainError("NODE_MAGNITUDE_LIMIT")
    return result


def _evaluate(tree: tuple, x: Decimal) -> Decimal:
    op = tree[0]
    if op == "one":
        return Decimal(1)
    if op == "var":
        return x
    return _safe_eml(_evaluate(tree[1], x), _evaluate(tree[2], x))


def _reference(operation: str, x: Decimal) -> Decimal:
    if operation == "e":
        return Decimal(1).exp()
    if operation == "exp":
        if abs(x) > MAX_EXPONENT_ARGUMENT:
            raise EMLDomainError("EXPONENT_ARGUMENT_LIMIT")
        return x.exp()
    if x <= 0:
        raise EMLDomainError("REAL_LOG_DOMAIN_REFUSED")
    return x.ln()


def witness(operation: str, samples: list[str]) -> dict:
    """Compare known identities on bounded real-domain samples, never train.

    A numerical match is an *observed test case*; it is not a Lean proof,
    analytic continuation, complex-branch correctness or signed attestation.
    """
    tree = compile_known(operation)
    spec = _structure(tree)
    if not isinstance(samples, list) or not 1 <= len(samples) <= MAX_SAMPLES:
        raise EMLDomainError("SAMPLE_COUNT_INVALID")
    # No user-provided numeric value, exception message, or model output is
    # included in the returned evidence record. Indices only.
    results = []
    for index, raw in enumerate(samples):
        try:
            x = _decimal_input(raw)
            with localcontext() as ctx:
                ctx.prec = 48
                ctx.Emax = 100
                ctx.Emin = -100
                observed = _evaluate(tree, x)
                expected = _reference(operation, x)
                if not expected.is_finite() or abs(expected) > MAX_MAGNITUDE:
                    raise EMLDomainError("REFERENCE_OUT_OF_RANGE")
                tolerance = _TOLERANCE * max(abs(expected), Decimal(1))
                delta = abs(observed - expected)
                if delta > tolerance:
                    raise EMLDomainError("NUMERICAL_MISMATCH")
            results.append({"index": index, "state": "NUMERIC_MATCH",
                            "absolute_error": str(delta), "tolerance": str(tolerance)})
        except EMLDomainError as exc:
            results.append({"index": index, "state": "REFUSED", "reason": exc.code})
        except DecimalException:
            results.append({"index": index, "state": "REFUSED", "reason": "DECIMAL_OPERATION_REFUSED"})
    passed = sum(x["state"] == "NUMERIC_MATCH" for x in results)
    return {
        "schema": SCHEMA,
        "paper": PAPER_ID,
        "paper_author_credit": "Andrzej Odrzywolek (2026)",
        "upstream_research_code": SOURCE_URL,
        "implementation": "INDEPENDENT_PYTHON_STD_LIB_OFFLINE",
        "operation": operation,
        "expression_rpn": spec["rpn"],
        "node_count": spec["nodes"],
        "eml_node_count": spec["operator_nodes"],
        "depth": spec["max_depth"],
        "tree_sha3_256": spec["tree_sha3_256"],
        "real_domain": "BOUNDED_REAL_ONLY",
        "numeric_precision_digits": 48,
        "state": ("SAMPLED_NUMERIC_MATCH" if passed == len(results)
                  else "PARTIAL_REFUSAL" if passed else "REFUSED"),
        "samples_total": len(results),
        "samples_matching": passed,
        "samples_refused": len(results) - passed,
        "observations": results,
        "complex_branch_evaluated": False,
        "extended_reals_evaluated": False,
        "all_elementary_functions_implemented": False,
        "analytic_identity_proven_here": False,
        "lean_proof_verified": False,
        "source_issuer_verified": False,
        "trained_model": False,
        "gpu_execution": False,
        "execution_authority": "NONE",
        "production_qualified": False,
        "scope": (
            "Paper-attributed numeric tests of three published EML identities "
            "on bounded real inputs; no universal compiler, branch-cut proof, "
            "symbolic-regression quality, signer verification or production authority."
        ),
    }


def fixture() -> dict:
    """Fixed public-only reproducibility example, not a benchmark claim."""
    return {
        "exp": witness("exp", ["-2", "0", "0.5", "2"]),
        "ln": witness("ln", ["0.125", "1", "2.5", "10"]),
        "e": witness("e", ["0"]),
    }


if __name__ == "__main__":
    print(json.dumps(fixture(), sort_keys=True, separators=(",", ":"), ensure_ascii=True))
