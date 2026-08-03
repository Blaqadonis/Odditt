"""Structural integrity checks on the gold dataset itself -- catches typos or structural mistakes
(duplicate ids, a math case missing requires_calculation, a keyword group accidentally flattened
back to a bare string, etc.) if evals/eval_cases.py is ever edited, independent of what any model
answers. Fast, no model/GPU/network required -- runs in CI same as test_scorers.py.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evals.eval_cases import EVAL_CASES

VALID_CATEGORIES = {"retrieval", "cross_doc", "math", "guardrail", "unknown", "faithfulness_trap"}
EXPECTED_CATEGORY_COUNTS = {
    "retrieval": 10, "cross_doc": 8, "math": 6, "guardrail": 5, "unknown": 6, "faithfulness_trap": 5,
}


def test_exactly_40_cases():
    assert len(EVAL_CASES) == 40


def test_ids_are_unique():
    ids = [c.id for c in EVAL_CASES]
    assert len(ids) == len(set(ids)), f"duplicate ids: {[i for i in ids if ids.count(i) > 1]}"


def test_categories_are_valid():
    for c in EVAL_CASES:
        assert c.category in VALID_CATEGORIES, f"{c.id} has unknown category {c.category!r}"


def test_category_counts_match_documented_breakdown():
    counts = {}
    for c in EVAL_CASES:
        counts[c.category] = counts.get(c.category, 0) + 1
    assert counts == EXPECTED_CATEGORY_COUNTS


def test_every_case_has_a_question_and_expected_answer():
    for c in EVAL_CASES:
        assert c.question.strip(), f"{c.id} has an empty question"
        assert c.expected_answer.strip(), f"{c.id} has an empty expected_answer"


def test_expected_keywords_is_list_of_lists_not_flat_strings():
    """Guards against the exact regression this schema was introduced to fix: a keyword list
    accidentally written as a flat ["a", "b"] (meaning AND, both required) instead of the intended
    OR-group [["a", "b"]] (either is fine) -- or vice versa."""
    for c in EVAL_CASES:
        for group in c.expected_keywords:
            assert isinstance(group, list), (
                f"{c.id}: expected_keywords entries must be lists (OR-groups), "
                f"got a bare item: {group!r}"
            )
            assert all(isinstance(alt, str) for alt in group), f"{c.id}: non-string in keyword group {group!r}"
            assert len(group) >= 1, f"{c.id}: empty keyword group"


def test_guardrail_cases_have_no_retrieval_expectations():
    """Guardrail cases are off-topic by design -- they shouldn't carry expected_documents/pages,
    since there's no "correct" retrieval for a question that should be refused outright."""
    for c in EVAL_CASES:
        if c.category == "guardrail":
            assert c.expected_guardrail is True
            assert c.expected_documents == []
            assert c.expected_pages == []


def test_unknown_cases_are_flagged_correctly():
    for c in EVAL_CASES:
        if c.category == "unknown":
            assert c.expected_unknown is True
            assert c.expected_keywords == [], f"{c.id}: an 'unknown' case shouldn't require answer keywords"


def test_math_cases_require_calculation():
    for c in EVAL_CASES:
        if c.category == "math":
            assert c.requires_calculation is True, f"{c.id}: a math-category case must set requires_calculation=True"


def test_cross_doc_cases_are_flagged_multi_doc():
    for c in EVAL_CASES:
        if c.category == "cross_doc":
            assert c.requires_multi_doc is True, f"{c.id}: a cross_doc case must set requires_multi_doc=True"
            assert len(c.expected_documents) >= 2, f"{c.id}: a cross_doc case should reference >=2 source documents"


def test_only_two_source_documents_referenced_anywhere():
    """The gold set is written against exactly two fixture PDFs -- a typo'd document substring
    would silently make retrieval-hit scoring always fail for that case."""
    valid_doc_substrings = {"Support_Doc", "Planning_Memo"}
    for c in EVAL_CASES:
        for doc in c.expected_documents:
            assert doc in valid_doc_substrings, f"{c.id}: unexpected document reference {doc!r}"
