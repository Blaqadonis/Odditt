"""The gold evaluation dataset -- extracted from the notebook's Section 8.1/8.2 cells.

40 hand-verified questions against the two mock PDFs in evals/fixtures/, across 6 categories.
Every expected answer and page number was checked against the actual PDF text, not written from
memory. See EvaluationCase docstring for how expected_keywords' OR-group structure works.
"""
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class EvaluationCase:
    id: str                                  # e.g. "RET-01"
    question: str
    category: str                            # retrieval | cross_doc | math | guardrail | unknown | faithfulness_trap
    expected_answer: str                     # short reference answer, human-written
    # List of OR-groups: each inner list is alternative phrasings of ONE fact -- any single match
    # in a group counts as that group satisfied. Multiple groups are AND'd together (all facts
    # must be present). e.g. [["1.4 million", "$1.4"]] = one fact, two acceptable phrasings.
    # e.g. [["substantive"], ["cycle count"], ["aging"]] = three distinct facts, all required.
    expected_keywords: List[List[str]] = field(default_factory=list)
    expected_documents: List[str] = field(default_factory=list)  # substrings of source filename
    expected_pages: List[int] = field(default_factory=list)      # 1-indexed page numbers
    difficulty: str = "medium"
    requires_reasoning: bool = False
    requires_calculation: bool = False
    requires_multi_doc: bool = False
    expected_guardrail: bool = False          # True => model should refuse (off-topic)
    expected_unknown: bool = False            # True => model should say "not in the document"
    notes: Optional[str] = None


# expected_keywords is a list of OR-groups: [["1.4 million", "$1.4"]] means "this one fact,
# either phrasing is fine" (previously both were required in an earlier version of this file,
# which silently failed on cases where the model used only one valid phrasing).
EVAL_CASES: List[EvaluationCase] = [

    # ---------------- Category A: single-document retrieval (10) ----------------
    EvaluationCase("RET-01", "What method does Nimbus use to value merchandise inventory?",
        "retrieval", "Lower of cost or net realizable value, using the weighted-average cost method.",
        [["weighted-average", "weighted average"], ["net realizable value"]], ["Support_Doc"], [2], "easy"),

    EvaluationCase("RET-02", "What dollar threshold requires Chief Financial Officer approval on a purchase requisition?",
        "retrieval", "$250,000", [["250,000"]], ["Support_Doc"], [3], "easy"),

    EvaluationCase("RET-03", "What price tolerance triggers manual review in the three-way match process?",
        "retrieval", "2% price tolerance", [["2%"]], ["Support_Doc"], [3], "medium"),

    EvaluationCase("RET-04", "What was Nimbus's weighted-average remaining lease term as of December 31, 2025?",
        "retrieval", "6.4 years", [["6.4"]], ["Support_Doc"], [3], "easy"),

    EvaluationCase("RET-05", "What annual base rent did Nimbus pay Halden Properties LLC in fiscal 2025?",
        "retrieval", "$1.4 million", [["1.4 million", "$1.4"]], ["Support_Doc"], [3], "easy"),

    EvaluationCase("RET-06", "What dollar amount requires dual-signature release for a payment?",
        "retrieval", "$100,000", [["100,000"]], ["Support_Doc"], [4], "easy"),

    EvaluationCase("RET-07", "What accrual did Nimbus record for the California wage-and-hour class action?",
        "retrieval", "$2.1 million", [["2.1 million", "$2.1"]], ["Support_Doc"], [5], "easy"),

    EvaluationCase("RET-08", "What is Nimbus's planning materiality for the FY2025 audit?",
        "retrieval", "$3,400,000", [["3,400,000", "3.4 million"]], ["Planning_Memo"], [2], "easy"),

    EvaluationCase("RET-09", "What is the trivial threshold (SAD) used in the FY2025 audit?",
        "retrieval", "$170,000", [["170,000"]], ["Planning_Memo"], [2], "medium"),

    EvaluationCase("RET-10", "When does interim fieldwork (controls testing) take place?",
        "retrieval", "September 15-26, 2025", [["September 15"], ["26, 2025"]], ["Planning_Memo"], [4], "easy"),

    # ---------------- Category B: cross-document / multi-hop (8) ----------------
    EvaluationCase("XDOC-01",
        "What amount did Nimbus pay for store-design consulting to a board member's family firm, and does the Planning Memo flag any additional related-party fraud risk beyond that disclosure?",
        "cross_doc", "$185,000 consulting fee; the Planning Memo states no additional related-party fraud risk factors were identified beyond what's already disclosed.",
        [["185,000"], ["no"], ["fraud risk"]], ["Support_Doc", "Planning_Memo"], [3, 2], "hard", requires_multi_doc=True),

    EvaluationCase("XDOC-02",
        "Compare the litigation accrual in the Accounting Policies narrative to the loss range discussed in the Planning Memo's significant risks section.",
        "cross_doc", "$2.1 million accrual, with a reasonably possible loss range of $1.5 million to $4.0 million — the same figures appear in both documents.",
        [["2.1 million"], ["1.5 million"], ["4.0 million"]], ["Support_Doc", "Planning_Memo"], [5, 3], "medium", requires_multi_doc=True),

    EvaluationCase("XDOC-03",
        "What audit approach does the Planning Memo assign to inventory, and what reserve methodology in the Accounting Policies document does that connect to?",
        "cross_doc", "Substantive approach relying on cycle counts; connects to the aging-based reserve methodology (25%/60%/100% by age band).",
        [["substantive"], ["cycle count"], ["aging"]], ["Planning_Memo", "Support_Doc"], [4, 2], "hard", requires_multi_doc=True),

    EvaluationCase("XDOC-04",
        "The Planning Memo treats revenue recognition as a fraud risk area — which specific policy does this relate to, and how much could a 2-point assumption change affect revenue?",
        "cross_doc", "The Nimbus Perks loyalty program breakage estimate; a 2-percentage-point change would affect revenue by approximately $610,000.",
        [["breakage"], ["610,000"]], ["Planning_Memo", "Support_Doc"], [3, 2], "hard", requires_multi_doc=True, requires_reasoning=True),

    EvaluationCase("XDOC-05",
        "What lease discount rate does the Planning Memo say will be evaluated, and where is that same rate first disclosed?",
        "cross_doc", "7.1% weighted-average discount rate, first disclosed in the Accounting Policies narrative's Leases section.",
        [["7.1%"]], ["Planning_Memo", "Support_Doc"], [3, 3], "medium", requires_multi_doc=True),

    EvaluationCase("XDOC-06", "How many retail stores does Nimbus operate, per both documents?",
        "cross_doc", "142 stores — consistent across both documents.",
        [["142"]], ["Support_Doc", "Planning_Memo"], [2, 2], "easy", requires_multi_doc=True),

    EvaluationCase("XDOC-07",
        "What audit approach is assigned to Accounts Payable / Accrued Liabilities, and what procure-to-pay control supports that rationale?",
        "cross_doc", "Combined approach, citing effective prior-year procure-to-pay controls such as the three-way match and the Delegation of Authority matrix.",
        [["combined"], ["three-way match", "three way match"]], ["Planning_Memo", "Support_Doc"], [4, 3], "hard", requires_multi_doc=True),

    EvaluationCase("XDOC-08",
        "Does the Planning Memo identify any fraud risk tied to the Halden Properties HQ lease beyond what's already disclosed?",
        "cross_doc", "No — the memo explicitly states no additional related-party fraud risk factors were identified.",
        [["no"], ["related-party", "related party"]], ["Planning_Memo", "Support_Doc"], [2, 3], "medium", requires_multi_doc=True),

    # ---------------- Category C: math / calculation (6) ----------------
    EvaluationCase("MATH-01", "What is the difference between planning materiality and performance materiality?",
        "math", "$850,000", [["850,000"]], ["Planning_Memo"], [2], "medium", requires_calculation=True),

    EvaluationCase("MATH-02", "What percentage of planning materiality is the trivial threshold?",
        "math", "5%", [["5%"]], ["Planning_Memo"], [2], "medium", requires_calculation=True),

    EvaluationCase("MATH-03", "By how much did the new revolving credit facility exceed the one it replaced?",
        "math", "$15,000,000", [["15,000,000", "15 million"]], ["Support_Doc"], [5], "easy", requires_calculation=True),

    EvaluationCase("MATH-04", "What is the midpoint of the reasonably possible litigation loss range?",
        "math", "$2.75 million", [["2.75"]], ["Support_Doc", "Planning_Memo"], [5, 3], "medium", requires_calculation=True),

    EvaluationCase("MATH-05",
        "How many days elapsed between the subsequent-events evaluation date (Feb 14, 2026) and the target report issuance date (Feb 20, 2026)?",
        "math", "6 days", [["6"]], ["Support_Doc", "Planning_Memo"], [5, 4], "hard", requires_calculation=True),

    EvaluationCase("MATH-06",
        "How many weeks elapsed between the start of interim fieldwork (Sept 15, 2025) and the target report issuance date (Feb 20, 2026)?",
        "math", "approximately 22.6 weeks", [["22", "23"]], ["Planning_Memo"], [4], "hard", requires_calculation=True),

    # ---------------- Category D: guardrail (5) ----------------
    EvaluationCase("GRD-01", "What's the capital of France?", "guardrail", "(refusal expected)",
        [], [], [], "easy", expected_guardrail=True),
    EvaluationCase("GRD-02", "What's the weather like today?", "guardrail", "(refusal expected)",
        [], [], [], "easy", expected_guardrail=True),
    EvaluationCase("GRD-03", "Can you give me a recipe for pasta carbonara?", "guardrail", "(refusal expected)",
        [], [], [], "easy", expected_guardrail=True),
    EvaluationCase("GRD-04", "Tell me a joke.", "guardrail", "(refusal expected)",
        [], [], [], "easy", expected_guardrail=True),
    EvaluationCase("GRD-05", "What model are you running on?", "guardrail", "(refusal expected)",
        [], [], [], "medium", expected_guardrail=True),

    # ---------------- Category E: unknown / absent from documents (6) ----------------
    EvaluationCase("UNK-01", "Who is Nimbus's Chief Executive Officer?", "unknown", "(not in the document)",
        [], ["Support_Doc", "Planning_Memo"], [], "medium", expected_unknown=True),
    EvaluationCase("UNK-02", "What is Nimbus's dividend policy?", "unknown", "(not in the document)",
        [], ["Support_Doc", "Planning_Memo"], [], "medium", expected_unknown=True),
    EvaluationCase("UNK-03", "What was Nimbus's total revenue for fiscal 2025?", "unknown", "(not in the document)",
        [], ["Support_Doc", "Planning_Memo"], [], "medium", expected_unknown=True),
    EvaluationCase("UNK-04", "How many employees does Nimbus have?", "unknown", "(not in the document)",
        [], ["Support_Doc", "Planning_Memo"], [], "easy", expected_unknown=True),
    EvaluationCase("UNK-05", "What audit opinion (unqualified, qualified, etc.) was issued?", "unknown", "(not in the document)",
        [], ["Support_Doc", "Planning_Memo"], [], "hard", expected_unknown=True),
    EvaluationCase("UNK-06", "Who is the Engagement Partner named on this audit?", "unknown", "(not in the document)",
        [], ["Planning_Memo"], [], "medium", expected_unknown=True),

    # ---------------- Category F: faithfulness traps (5) ----------------
    EvaluationCase("TRAP-01", "What threshold specifically requires CFO approval (not director or VP-level approval)?",
        "faithfulness_trap", "$250,000", [["250,000"]], ["Support_Doc"], [3], "hard",
        notes="Distractors in same passage: $10,000 (director), $75,000 (VP Finance)."),
    EvaluationCase("TRAP-02", "What is the price tolerance specifically (not the quantity tolerance) in the three-way match?",
        "faithfulness_trap", "2%", [["2%"]], ["Support_Doc"], [3], "hard",
        notes="Distractor: 5% is the quantity tolerance in the same sentence."),
    EvaluationCase("TRAP-03", "What is performance materiality specifically (not planning materiality)?",
        "faithfulness_trap", "$2,550,000", [["2,550,000"]], ["Planning_Memo"], [2], "hard",
        notes="Distractor: $3,400,000 is planning materiality, in the same table."),
    EvaluationCase("TRAP-04", "What was the size of the credit facility that was replaced (not the new one)?",
        "faithfulness_trap", "$25 million", [["25 million", "25,000,000"]], ["Support_Doc"], [5], "hard",
        notes="Distractor: $40 million is the new facility, in the same sentence."),
    EvaluationCase("TRAP-05", "Which bank led the syndicate for the new credit facility?",
        "faithfulness_trap", "Meridian National Bank", [["Meridian"]], ["Support_Doc"], [5], "medium",
        notes="Tests fabrication -- no other bank name appears anywhere in either document."),
]
