"""Entry point: python -m evals.run --build-dataset | --retrieval | --generation | --all

Run from the securerag/ directory (needs a populated ChromaDB and, for dataset/generation, GROQ_API_KEY).
Writes evals/results/<timestamp>.json and evals/results/latest.md.
"""
import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results"


def main() -> None:
    p = argparse.ArgumentParser(description="SecureRAG evaluation harness")
    p.add_argument("--build-dataset", action="store_true", help="(re)generate evals/data/retrieval_set.jsonl")
    p.add_argument("--retrieval", action="store_true", help="BM25 / dense / hybrid / rerank ablation")
    p.add_argument("--generation", action="store_true", help="citation + abstention checks (calls Groq)")
    p.add_argument("--all", action="store_true")
    args = p.parse_args()
    if not (args.build_dataset or args.retrieval or args.generation or args.all):
        p.error("choose at least one of --build-dataset --retrieval --generation --all")

    logging.basicConfig(level=logging.WARNING)
    from evals import build_dataset, generation_eval, retrieval_eval
    from rag.vectorstore import get_counts_by_source

    counts = get_counts_by_source()
    if counts["total"] == 0:
        raise SystemExit("ChromaDB is empty — run `python data/ingest.py` first.")

    if args.build_dataset or args.all:
        build_dataset.build()

    from core.config import settings

    report: dict = {"timestamp": datetime.now(timezone.utc).isoformat(), "corpus": counts}
    md = [f"## SecureRAG evaluation — {report['timestamp'][:19]}Z",
          f"Corpus: {counts['cve']} CVEs + {counts['owasp']} OWASP chunks", ""]

    if args.retrieval or args.generation or args.all:
        dataset = build_dataset.load_dataset()

    if args.retrieval or args.all:
        report["retrieval"] = retrieval_eval.run_retrieval_eval(dataset)
        md += [retrieval_eval.to_markdown(report["retrieval"]), ""]

    if args.generation or args.all:
        report["generation"] = generation_eval.run_generation_eval(dataset)
        report["generation"]["llm_model"] = settings.groq_model
        md += [f"LLM for generation checks: `{settings.groq_model}`", "", generation_eval.to_markdown(report["generation"]), ""]

    if len(report) > 2:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        (RESULTS_DIR / f"{stamp}.json").write_text(json.dumps(report, indent=1))
        (RESULTS_DIR / "latest.md").write_text("\n".join(md))
        print("\n".join(md))
        print(f"\nSaved evals/results/{stamp}.json and evals/results/latest.md")


if __name__ == "__main__":
    main()
