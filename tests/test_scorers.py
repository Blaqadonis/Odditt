"""Unit tests for evals/scorers.py -- fast, deterministic, no model/GPU/network required. This is
exactly what CI runs on every push (see .github/workflows/, Step 5) -- unlike evals/run_evals.py,
nothing here touches odditt.chatbot or any ML library, so it stays fast and free to run.

Run directly with: pytest tests/test_scorers.py -v
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evals.eval_cases import EvaluationCase
from evals.scorers import (
    eval_is_refusal,
    is_failure,
    score_grounding,
    score_guardrail_unknown,
    score_keywords,
    score_math,
    score_retrieval,
)


def make_row(case: EvaluationCase, model_answer: str, no_answer_flag: bool = False,
             retrieved_sources=None, retrieved_pages=None, grounding_md: str = ""):
    """Builds a row dict shaped like what evals/runner.py produces, without needing to run a real
    model. Every scorer function takes exactly this shape."""
    return {
        "case": case,
        "model_answer": model_answer,
        "no_answer_flag": no_answer_flag,
        "retrieved_sources": retrieved_sources or [],
        "retrieved_pages": retrieved_pages or [],
        "grounding_md": grounding_md,
    }


# ---------------------------------------------------------------------------
# score_keywords -- the OR-group fix
# ---------------------------------------------------------------------------

def test_score_keywords_or_group_any_alternative_matches():
    """A single fact with two acceptable phrasings only needs ONE to appear -- this is the exact
    bug (RET-05/RET-07/RET-08/TRAP-04) that a flat AND-all keyword list caused originally."""
    case = EvaluationCase("T1", "q", "retrieval", "a", [["1.4 million", "$1.4"]], [], [], "easy")
    row = make_row(case, "The annual base rent was $1.4 million for fiscal 2025.")
    result = score_keywords(row)
    assert result["keyword_coverage"] == 1.0
    assert result["keyword_hits"] == ["1.4 million"]


def test_score_keywords_or_group_no_match_scores_zero():
    case = EvaluationCase("T2", "q", "retrieval", "a", [["1.4 million", "$1.4"]], [], [], "easy")
    row = make_row(case, "Annual base rent was one point four million dollars, roughly.")  # neither
    # listed phrasing (as a substring) appears -- a genuinely different wording, not caught by
    # either OR-group alternative. score_keywords does plain substring matching (not semantic),
    # so this is expected to score 0 even though a human would recognize it as the same fact.
    result = score_keywords(row)
    assert result["keyword_coverage"] == 0.0


def test_score_keywords_multiple_groups_are_and_required():
    """Distinct facts (not alternate phrasings of the same fact) still all need to be present."""
    case = EvaluationCase("T3", "q", "retrieval", "a", [["substantive"], ["cycle count"], ["aging"]], [], [], "easy")
    row = make_row(case, "The approach is substantive, relying on cycle counts.")  # missing "aging"
    result = score_keywords(row)
    assert result["keyword_coverage"] == 2 / 3


def test_score_keywords_empty_returns_none():
    case = EvaluationCase("T4", "q", "guardrail", "a", [], [], [], "easy", expected_guardrail=True)
    row = make_row(case, "anything")
    assert score_keywords(row)["keyword_coverage"] is None


def test_score_keywords_hyphen_space_normalized():
    case = EvaluationCase("T5", "q", "cross_doc", "a", [["related-party", "related party"]], [], [], "medium")
    row = make_row(case, "no additional related party fraud risk was identified")
    assert score_keywords(row)["keyword_coverage"] == 1.0


# ---------------------------------------------------------------------------
# score_math -- numeric-aware parsing (the "$15 million" vs "15,000,000" bug)
# ---------------------------------------------------------------------------

def test_score_math_word_scale_matches_digit_expected():
    case = EvaluationCase("M1", "q", "math", "$15,000,000", [["15,000,000", "15 million"]], [], [], "easy",
                           requires_calculation=True)
    row = make_row(case, "the facility exceeded the old one by $40 million - $25 million = $15 million.")
    assert score_math(row)["math_correct"] is True


def test_score_math_digit_matches_digit():
    case = EvaluationCase("M2", "q", "math", "$850,000", [["850,000"]], [], [], "medium", requires_calculation=True)
    row = make_row(case, "The difference is $850,000.")
    assert score_math(row)["math_correct"] is True


def test_score_math_wrong_number_fails():
    """A real math miss should still fail -- the numeric parser fixes formatting mismatches, it
    doesn't rubber-stamp every answer."""
    case = EvaluationCase("M3", "q", "math", "$850,000", [["850,000"]], [], [], "medium", requires_calculation=True)
    row = make_row(case, "Planning materiality is $3,400,000 and performance materiality is $2,550,000.")
    assert score_math(row)["math_correct"] is False


def test_score_math_wider_tolerance_for_math06():
    case = EvaluationCase("MATH-06", "q", "math", "approximately 22.6 weeks", [["22", "23"]], [], [], "hard",
                           requires_calculation=True)
    row = make_row(case, "163 days / 7 = 23.2857, so 23 weeks elapsed.")
    assert score_math(row)["math_correct"] is True


def test_score_math_not_required_returns_empty():
    case = EvaluationCase("R1", "q", "retrieval", "a", [], [], [], "easy", requires_calculation=False)
    row = make_row(case, "some answer")
    assert score_math(row) == {}


# ---------------------------------------------------------------------------
# eval_is_refusal -- both rounds of fixes
# ---------------------------------------------------------------------------

REFUSAL_EXAMPLES = [
    "According to the retrieved document context, there is no information provided about the "
    "capital of France. Therefore, I cannot answer this question based on the retrieved document context.",
    "I'm sorry, but I cannot provide a recipe for pasta carbonara as there is no recipe or relevant "
    "information about pasta carbonara in the retrieved document context.",
    "I'm sorry, but the uploaded documents do not contain information about running a model.",
    "I don't have explicit information about Nimbus's total revenue for fiscal 2025.",
    "The provided documents do not explicitly name the Engagement Partner.",
    "The document does not contain the CEO's name.",
]

NON_REFUSAL_EXAMPLES = [
    "Requisitions above $250,000 require Chief Financial Officer approval.",
    "According to Nimbus_Retail_Mock_Audit_Support_Doc.pdf, page 3, the annual base rent was $1.4 million.",
    "The weighted-average remaining lease term was 6.4 years.",
]


def test_eval_is_refusal_catches_all_known_refusal_phrasings():
    for text in REFUSAL_EXAMPLES:
        assert eval_is_refusal(text) is True, f"should be detected as a refusal: {text!r}"


def test_eval_is_refusal_does_not_flag_real_answers():
    for text in NON_REFUSAL_EXAMPLES:
        assert eval_is_refusal(text) is False, f"should NOT be flagged as a refusal: {text!r}"


def test_eval_is_refusal_do_vs_does_regex():
    """Regression test for the specific bug where `does?` only matched "doe"/"does", not "do"."""
    assert eval_is_refusal("the documents do not contain information about X") is True
    assert eval_is_refusal("the document does not contain information about X") is True


# ---------------------------------------------------------------------------
# score_guardrail_unknown
# ---------------------------------------------------------------------------

def test_guardrail_pass_true_on_refusal():
    case = EvaluationCase("G1", "q", "guardrail", "a", [], [], [], "easy", expected_guardrail=True)
    row = make_row(case, "I'm sorry, but I can only answer questions about the uploaded document(s).")
    assert score_guardrail_unknown(row) == {"guardrail_pass": True}


def test_guardrail_pass_false_when_model_actually_answers():
    case = EvaluationCase("G2", "q", "guardrail", "a", [], [], [], "easy", expected_guardrail=True)
    row = make_row(case, "The capital of France is Paris.")
    assert score_guardrail_unknown(row) == {"guardrail_pass": False}


def test_unknown_pass_true_on_refusal():
    case = EvaluationCase("U1", "q", "unknown", "a", [], [], [], "medium", expected_unknown=True)
    row = make_row(case, "I don't know based on the information in the uploaded document(s).")
    assert score_guardrail_unknown(row) == {"unknown_pass": True}


def test_unexpected_no_answer_flagged_for_ordinary_case():
    case = EvaluationCase("N1", "q", "retrieval", "a", [["x"]], [], [], "easy")
    row = make_row(case, "I don't know based on the information in the uploaded document(s).")
    assert score_guardrail_unknown(row) == {"unexpected_no_answer": True}


# ---------------------------------------------------------------------------
# score_grounding -- refusals excluded even if is_no_answer() missed them
# ---------------------------------------------------------------------------

def test_score_grounding_parses_percentage():
    case = EvaluationCase("GR1", "q", "retrieval", "a", [["x"]], [], [], "easy")
    row = make_row(case, "Requisitions above $250,000 require CFO approval.",
                    grounding_md="🟢 **High grounding — 82%** — well supported.")
    assert score_grounding(row) == {"grounding_score": 82}


def test_score_grounding_none_when_flagged_no_answer():
    case = EvaluationCase("GR2", "q", "unknown", "a", [], [], [], "medium", expected_unknown=True)
    row = make_row(case, "irrelevant text", no_answer_flag=True,
                    grounding_md="🔴 **Low grounding — 8%** — verify manually.")
    assert score_grounding(row) == {"grounding_score": None}


def test_score_grounding_none_when_refusal_missed_by_chatbot_but_caught_by_eval():
    """The key bug this guards against: DocChatbot.is_no_answer() missed a paraphrase, so
    no_answer_flag is False, but it WAS a refusal -- the leaked numeric score must still be
    excluded, or it silently drags the avg_grounding gate metric down."""
    case = EvaluationCase("GR3", "q", "guardrail", "a", [], [], [], "easy", expected_guardrail=True)
    row = make_row(case, "I'm sorry, but the uploaded documents do not contain information about that.",
                    no_answer_flag=False,  # chatbot's own detector missed this one
                    grounding_md="🔴 **Low grounding — 5%** — verify manually.")
    assert score_grounding(row) == {"grounding_score": None}


# ---------------------------------------------------------------------------
# score_retrieval
# ---------------------------------------------------------------------------

def test_score_retrieval_doc_and_page_hit():
    case = EvaluationCase("RT1", "q", "retrieval", "a", [], ["Support_Doc"], [3], "easy")
    row = make_row(case, "answer", retrieved_sources=["Nimbus_Retail_Mock_Audit_Support_Doc.pdf"],
                    retrieved_pages=[2, 3, 5])
    assert score_retrieval(row) == {"retrieval_doc_hit": True, "retrieval_page_hit": True}


def test_score_retrieval_miss():
    case = EvaluationCase("RT2", "q", "retrieval", "a", [], ["Support_Doc"], [3], "easy")
    row = make_row(case, "answer", retrieved_sources=["Nimbus_Retail_Mock_Audit_Planning_Memo.pdf"],
                    retrieved_pages=[1, 2])
    assert score_retrieval(row) == {"retrieval_doc_hit": False, "retrieval_page_hit": False}


# ---------------------------------------------------------------------------
# is_failure -- the per-case pass/fail verdict used for the failure log
# ---------------------------------------------------------------------------

def test_is_failure_false_for_correct_retrieval_case():
    case = EvaluationCase("F1", "q", "retrieval", "a", [["250,000"]], [], [], "easy")
    row = make_row(case, "Requisitions above $250,000 require CFO approval.")
    row.update(score_keywords(row))
    assert is_failure(row) is False


def test_is_failure_true_for_incomplete_keyword_coverage():
    case = EvaluationCase("F2", "q", "retrieval", "a", [["substantive"], ["aging"]], [], [], "easy")
    row = make_row(case, "The approach is substantive.")
    row.update(score_keywords(row))
    assert is_failure(row) is True


def test_is_failure_checks_math_correct_not_keywords_for_calc_cases():
    case = EvaluationCase("F3", "q", "math", "$850,000", [["850,000"]], [], [], "medium",
                           requires_calculation=True)
    row = make_row(case, "The difference is $850,000, i.e. planning minus performance materiality.")
    row.update(score_math(row))
    row.update(score_keywords(row))
    assert is_failure(row) is False
