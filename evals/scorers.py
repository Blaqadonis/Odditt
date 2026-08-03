"""Deterministic scorers -- no API calls, fully reproducible. Extracted from the notebook's
Section 8.4 cell, including the two rounds of fixes applied after the first real eval runs:

  1. score_keywords: expected_keywords is a list of OR-groups (see eval_cases.py), not a flat
     AND-all list -- alternate phrasings of the same fact ("$1.4" / "1.4 million") only need one
     to appear, not both.
  2. score_math: parses "$15 million" / "15,000,000" / "6 days" as actual numbers with a
     tolerance check, instead of exact digit-string matching.
  3. eval_is_refusal(): a superset of DocChatbot.is_no_answer()'s phrase list, used only for eval
     scoring. Round 1 added literal phrases Phi used that the shipped is_no_answer() missed
     ("cannot provide", "unable to answer this question", ...). Round 2 added regex patterns
     after Qwen (more linguistically varied than Phi) produced correct refusals in shapes no
     literal phrase list could keep up with ("do/does not contain/name/state...", "don't have
     explicit information...", "not explicitly named/stated/disclosed...").

DocChatbot.is_no_answer() itself is untouched -- that's product code, not eval code.
"""
import re

_NUM_PATTERN = re.compile(r"\$?\s*([\d,]+(?:\.\d+)?)\s*(million|billion|thousand)?", re.IGNORECASE)

_EVAL_REFUSAL_PHRASES = (
    "only answer questions about",
    "don't know based on the information",
    "do not know based on the information",
    "don't have the information",
    "do not have the information",
    "does not contain information",
    "context does not contain",
    "not contain any information",
    "not provided in the given document",
    "not provided in the retrieved document",
    "no information about",
    "no information regarding",
    "no information provided about",
    "there is no information provided",
    "i cannot answer this question",
    "i am unable to answer this question",
    "i'm unable to answer this question",
    "cannot provide",
    "cannot answer this question",
    "unable to answer this question",
)

# Regex patterns that generalize the SHAPE of a refusal rather than one exact wording -- added
# after literal phrases kept missing new models' paraphrasings one at a time.
_EVAL_REFUSAL_PATTERNS = (
    # "do/does not (explicitly) contain/name/state/mention/include/disclose/specify ..."
    re.compile(r"\bdo(?:es)? not (?:explicitly )?(?:contain|name|state|mention|include|disclose|specify)\b"),
    # "don't/doesn't have (explicit/specific/enough) information ..."
    re.compile(r"\b(?:don'?t|does ?n'?t) have (?:explicit |specific |enough )?information\b"),
    # "not explicitly named/stated/disclosed/provided/mentioned" (passive phrasing)
    re.compile(r"\bnot explicitly (?:named|stated|disclosed|provided|mentioned)\b"),
)


def eval_is_refusal(answer: str) -> bool:
    a = (answer or "").lower()
    if any(p in a for p in _EVAL_REFUSAL_PHRASES):
        return True
    return any(pat.search(a) for pat in _EVAL_REFUSAL_PATTERNS)


def _normalize(s: str) -> str:
    return s.lower().replace("-", " ").replace("  ", " ")


def score_retrieval(row) -> dict:
    case = row["case"]
    if not case.expected_documents:
        return {"retrieval_doc_hit": None, "retrieval_page_hit": None}
    doc_hit = any(
        any(exp_doc in src for exp_doc in case.expected_documents)
        for src in row["retrieved_sources"]
    )
    page_hit = any(p in case.expected_pages for p in row["retrieved_pages"]) if case.expected_pages else None
    return {"retrieval_doc_hit": doc_hit, "retrieval_page_hit": page_hit}


def score_keywords(row) -> dict:
    case = row["case"]
    if not case.expected_keywords:
        return {"keyword_coverage": None, "keyword_hits": []}
    answer_norm = _normalize(row["model_answer"])
    hits = []
    groups_hit = 0
    for group in case.expected_keywords:
        matched = next((alt for alt in group if _normalize(alt) in answer_norm), None)
        if matched:
            groups_hit += 1
            hits.append(matched)
    coverage = groups_hit / len(case.expected_keywords)
    return {"keyword_coverage": coverage, "keyword_hits": hits}


def score_guardrail_unknown(row) -> dict:
    case = row["case"]
    refusal = bool(row["no_answer_flag"]) or eval_is_refusal(row["model_answer"])
    if case.expected_guardrail:
        return {"guardrail_pass": refusal}
    if case.expected_unknown:
        return {"unknown_pass": refusal}
    return {"unexpected_no_answer": refusal}


def _extract_numbers(text: str):
    vals = []
    for m in _NUM_PATTERN.finditer(text or ""):
        num_str, scale = m.group(1), m.group(2)
        if not num_str or num_str.strip(",.") == "":
            continue
        try:
            val = float(num_str.replace(",", ""))
        except ValueError:
            continue
        if scale:
            scale = scale.lower()
            if scale.startswith("million"):
                val *= 1_000_000
            elif scale.startswith("billion"):
                val *= 1_000_000_000
            elif scale.startswith("thousand"):
                val *= 1_000
        vals.append(val)
    return vals


def score_math(row) -> dict:
    case = row["case"]
    if not case.requires_calculation:
        return {}
    expected_vals = _extract_numbers(case.expected_answer)
    if not expected_vals:
        return {"math_correct": None}
    target = expected_vals[0]
    answer_vals = _extract_numbers(row["model_answer"])
    # Wider tolerance for the "elapsed weeks" case, since "approximately 22.6 weeks" rounding to
    # 23 whole weeks is a reasonable answer, not an error.
    tol_pct = 0.05 if case.id == "MATH-06" else 0.01
    tol = max(tol_pct * abs(target), 0.5)
    match = any(abs(v - target) <= tol for v in answer_vals)
    return {"math_correct": match}


def score_grounding(row) -> dict:
    # If this was actually a refusal (even one DocChatbot.is_no_answer() missed), the numeric
    # grounding % it computed is meaningless -- there's no real "answer" to be grounded. Exclude
    # it from the grounding average rather than letting it drag the category score down.
    refusal = bool(row["no_answer_flag"]) or eval_is_refusal(row["model_answer"])
    if refusal:
        return {"grounding_score": None}
    m = re.search(r"(\d+)%", row["grounding_md"] or "")
    return {"grounding_score": int(m.group(1)) if m else None}


def score_row(row) -> dict:
    merged = {}
    for fn in (score_retrieval, score_keywords, score_guardrail_unknown, score_math, score_grounding):
        merged.update(fn(row))
    return merged


def is_failure(row) -> bool:
    """A single boolean verdict per case, used for the failure log and pass-count summaries."""
    case = row["case"]
    if case.expected_guardrail:
        return not row.get("guardrail_pass", False)
    if case.expected_unknown:
        return not row.get("unknown_pass", False)
    if case.requires_calculation:
        return not row.get("math_correct", False)
    cov = row.get("keyword_coverage")
    return (cov is not None and cov < 1.0) or bool(row.get("unexpected_no_answer"))
