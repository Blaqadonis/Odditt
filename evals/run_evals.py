"""CLI entrypoint for the full 40-case eval against a real model.

    pip install -r requirements-dev.txt
    python -m evals.run_evals --model microsoft/Phi-4-mini-instruct
    python -m evals.run_evals --model Qwen/Qwen2.5-3B-Instruct --output-dir results/

Not part of CI (see README) -- this needs a GPU and downloads/loads a real multi-GB model, same
as running Section 8 in the notebook did. CI instead runs the fast, model-free scorer tests in
tests/ on every push (Step 4/5).
"""
import argparse
import os
import sys

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures")
DEFAULT_PDFS = [
    os.path.join(FIXTURES_DIR, "Nimbus_Retail_Mock_Audit_Support_Doc.pdf"),
    os.path.join(FIXTURES_DIR, "Nimbus_Retail_Mock_Audit_Planning_Memo.pdf"),
]


def main():
    parser = argparse.ArgumentParser(description="Run Odditt's 40-case evaluation suite against a real model.")
    parser.add_argument("--model", default=None,
                         help="HF model id to evaluate (default: CONFIG['llm_model'], i.e. Phi-4-mini-instruct).")
    parser.add_argument("--pdfs", nargs="+", default=DEFAULT_PDFS,
                         help="PDF(s) to index and evaluate against (default: the two Nimbus mock fixtures).")
    parser.add_argument("--output-dir", default=".",
                         help="Where to write eval_results/eval_summary/eval_failures files (default: cwd).")
    parser.add_argument("--fail-under-gate", action="store_true", default=True,
                         help="Exit with a non-zero status if the deployment gate fails (default: on).")
    args = parser.parse_args()

    missing = [p for p in args.pdfs if not os.path.exists(p)]
    if missing:
        print("ERROR -- these PDFs were not found:")
        for m in missing:
            print(" -", m)
        sys.exit(2)

    # Imported lazily, after arg parsing: these pull in transformers/torch/etc., which is slow
    # and unnecessary if the user just ran --help or hit the missing-PDFs error above.
    from odditt.chatbot import DocChatbot
    from odditt.config import CONFIG
    from odditt.model_loader import load_embeddings, load_llm

    from .eval_cases import EVAL_CASES
    from .report import deployment_gate, write_report
    from .runner import run_evaluation
    from .scorers import score_row

    model_id = args.model or CONFIG["llm_model"]
    run_config = dict(CONFIG)
    run_config["llm_model"] = model_id

    print(f"Evaluating {model_id} against {len(EVAL_CASES)} cases, {len(args.pdfs)} PDF(s)...\n")

    embeddings = load_embeddings(run_config)
    _model, _tokenizer, _pipeline, llm = load_llm(model_id, run_config)
    chatbot = DocChatbot(run_config, embeddings, llm)

    raw_df = run_evaluation(chatbot, EVAL_CASES, args.pdfs)
    scores_df = raw_df.apply(score_row, axis=1, result_type="expand")
    results_df = raw_df.join(scores_df)

    os.makedirs(args.output_dir, exist_ok=True)
    write_report(results_df, output_dir=args.output_dir)

    print()
    gate_passed = deployment_gate(results_df)

    if args.fail_under_gate and not gate_passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
