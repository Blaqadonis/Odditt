"""Odditt evaluation framework -- gold questions, runner, scorers, and deployment gate.

Deliberately NOT re-exported at package level (no `from evals import run_evaluation` shortcut):
runner.py imports odditt.chatbot, which imports the full transformers/langchain/torch stack, and
eagerly re-exporting it here would mean even `python -m evals.run_evals --help` or an early
argument-validation error triggers that entire heavy import chain -- exactly what run_evals.py's
lazy imports (inside main(), after arg parsing) are trying to avoid. Import directly from each
submodule instead:

    from evals.eval_cases import EVAL_CASES, EvaluationCase      # no heavy deps
    from evals.scorers import score_row, is_failure               # no heavy deps
    from evals.report import summarize, write_report, deployment_gate, GATE_THRESHOLDS  # pandas only
    from evals.runner import run_evaluation                       # pulls in odditt.chatbot (heavy)

Run the full thing end-to-end with:

    python -m evals.run_evals --model microsoft/Phi-4-mini-instruct
"""
