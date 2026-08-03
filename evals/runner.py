"""Drives DocChatbot exactly the way the Gradio UI does -- through process_pdfs(pdf_files, query,
history) -- so the eval reflects real end-to-end behavior, not a shortcut around the
prompt/retrieval/guardrail logic. Extracted from the notebook's Section 8.3b cell.
"""
import os
import time
from typing import List

import pandas as pd

from odditt.chatbot import DocChatbot

from .eval_cases import EvaluationCase


def run_evaluation(chatbot: DocChatbot, cases: List[EvaluationCase], pdf_files: List[str],
                    verbose: bool = True) -> pd.DataFrame:
    rows = []
    for case in cases:
        t0 = time.time()
        history, gallery, grounding_md, _ = chatbot.process_pdfs(pdf_files, case.question, [])
        latency = time.time() - t0

        answer = history[-1]["content"] if history else ""
        retrieved = chatbot.last_retrieved_docs or []
        no_answer = chatbot.is_no_answer(answer)

        rows.append({
            "id": case.id,
            "category": case.category,
            "question": case.question,
            "expected_answer": case.expected_answer,
            "model_answer": answer,
            "no_answer_flag": no_answer,
            "retrieved_sources": [os.path.basename(d.metadata.get("source", "")) for d in retrieved],
            "retrieved_pages": [d.metadata.get("page", -1) + 1 for d in retrieved],  # back to 1-indexed
            "grounding_md": grounding_md,
            "latency_sec": round(latency, 2),
            "case": case,
        })
        if verbose:
            print(f"  [{case.id}] {case.category:<18} {latency:5.1f}s  -> {answer[:70]!r}")
    return pd.DataFrame(rows)
