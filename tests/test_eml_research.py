# SPDX-License-Identifier: Apache-2.0
"""Offline EML paper-reproduction contracts: real-domain only, no GPU or model."""

import copy

import pytest

from szl_formulas import LOCKED_PROVEN_COUNT, LOCKED_PROVEN_FORMULA_IDS
from szl_formulas import eml_research as eml


def test_published_eml_identity_tree_rpn_lengths():
    exp_spec = eml._structure(eml.compile_known("exp"))
    log_spec = eml._structure(eml.compile_known("ln"))
    e_spec = eml._structure(eml.compile_known("e"))
    assert exp_spec["rpn"] == "x1E"
    assert log_spec["rpn"] == "11xE1EE"  # Paper Eq. (5), Sec. 4.1
    assert e_spec["rpn"] == "11E"
    assert (log_spec["nodes"], log_spec["operator_nodes"], log_spec["max_depth"]) == (7, 3, 3)
    assert len({v["tree_sha3_256"] for v in (exp_spec, log_spec, e_spec)}) == 3


@pytest.mark.parametrize("operation,samples", [
    ("exp", ["-2", "-0.5", "0", "0.000001", "1", "2", "10"]),
    ("ln", ["0.00000001", "0.1", "0.125", "1", "1.0001", "2.5", "10", "100"]),
    ("e", ["0"]),
])
def test_positive_real_domain_numeric_fixtures_match_independent_decimal(operation, samples):
    result = eml.witness(operation, samples)
    assert result["state"] == "SAMPLED_NUMERIC_MATCH"
    assert result["samples_total"] == len(samples)
    assert result["samples_matching"] == len(samples)
    assert result["samples_refused"] == 0
    assert all(row["state"] == "NUMERIC_MATCH" for row in result["observations"])
    assert result["paper"] == "arxiv:2603.21852v2"
    assert result["real_domain"] == "BOUNDED_REAL_ONLY"
    assert not result["all_elementary_functions_implemented"]
    assert not result["analytic_identity_proven_here"]
    assert not result["lean_proof_verified"]
    assert not result["gpu_execution"] and not result["trained_model"]
    assert result["execution_authority"] == "NONE"


def test_known_paper_tree_result_does_not_use_real_log_of_negative():
    mixed = eml.witness("ln", ["0", "-1", "1", "-0.5"])
    assert mixed["state"] == "PARTIAL_REFUSAL"
    assert mixed["samples_total"] == 4
    assert mixed["samples_matching"] == 1
    assert mixed["samples_refused"] == 3
    assert [r.get("reason") for r in mixed["observations"]] == [
        "REAL_LOG_DOMAIN_REFUSED", "REAL_LOG_DOMAIN_REFUSED", None,
        "REAL_LOG_DOMAIN_REFUSED",
    ]
    assert mixed["complex_branch_evaluated"] is False
    assert mixed["extended_reals_evaluated"] is False


def test_numerical_overflow_refused_not_clamped_into_valid_identity():
    for operation, sample, code in [
        ("exp", "51", "EXPONENT_ARGUMENT_LIMIT"),
        ("exp", "1000", "INPUT_OUT_OF_RANGE"),
        ("ln", "1e-12", "NODE_MAGNITUDE_LIMIT"),
    ]:
        result = eml.witness(operation, [sample])
        assert result["state"] == "REFUSED"
        assert result["samples_refused"] == 1
        assert result["observations"][0]["reason"] == code
        assert result["observations"][0].get("absolute_error") is None


@pytest.mark.parametrize("sample", [
    "NaN", "Infinity", "-Infinity", " 1 ", "1_000", "0x12", "",
    ".1", "1.", "+2", "01", "true", "1e9999", "bad " * 12,
    1, 1.0, True, None, {}, [],
])
def test_invalid_input_formats_have_finite_refusal_code(sample):
    report = eml.witness("exp", [sample])
    assert report["state"] == "REFUSED"
    assert report["observations"][0]["reason"] == "INPUT_NOT_CANONICAL_DECIMAL"
    assert "Traceback" not in str(report)
    assert report["source_issuer_verified"] is False


@pytest.mark.parametrize("operation", ["sin", "cos", "sqrt", "all", "__import__", "", None, 1, True])
def test_no_claim_of_unimplemented_calculator_or_expression_eval(operation):
    with pytest.raises(eml.EMLDomainError) as ex:
        eml.witness(operation, ["1"])
    assert ex.value.code == "OPERATION_UNSUPPORTED"


@pytest.mark.parametrize("samples", [[], ["1"] * 33, {}, "1", None, 1])
def test_sample_bound_fails_closed(samples):
    with pytest.raises(eml.EMLDomainError) as ex:
        eml.witness("ln", samples)
    assert ex.value.code == "SAMPLE_COUNT_INVALID"


def test_tree_grammar_invalid_operations_and_extra_arguments_fail():
    for node in (
        (), ("one", 1), ("var", "y"), ("eml", ("one",)),
        ("eval", ("one",), ("one",)), ("eml", "x", ("one",)),
    ):
        with pytest.raises(eml.EMLDomainError):
            eml._structure(node)


def test_tree_depth_and_size_bounds():
    root = ("one",)
    for _ in range(10):
        root = ("eml", root, ("one",))
    with pytest.raises(eml.EMLDomainError) as ex:
        eml._structure(root)
    assert ex.value.code in {"TREE_DEPTH_LIMIT", "TREE_NODE_LIMIT"}


def test_no_minting_of_locked_formula_status():
    before = copy.deepcopy(EML_LOCKED_MARKER)
    eml.witness("ln", ["1.5", "2"])
    assert EML_LOCKED_MARKER == before
    assert LOCKED_PROVEN_COUNT == 8
    assert LOCKED_PROVEN_FORMULA_IDS == frozenset(
        {"F1", "F4", "F7", "F11", "F12", "F18", "F19", "F22"}
    )


def test_fixture_repeated_is_deterministic_not_authenticating():
    one = eml.fixture()
    two = eml.fixture()
    assert one == two
    assert set(one) == {"exp", "ln", "e"}
    assert all(r["source_issuer_verified"] is False for r in one.values())
    assert all(r["production_qualified"] is False for r in one.values())
    assert all(len(r["tree_sha3_256"]) == 64 for r in one.values())


EML_LOCKED_MARKER = {
    "count": LOCKED_PROVEN_COUNT,
    "ids": tuple(sorted(LOCKED_PROVEN_FORMULA_IDS)),
}
