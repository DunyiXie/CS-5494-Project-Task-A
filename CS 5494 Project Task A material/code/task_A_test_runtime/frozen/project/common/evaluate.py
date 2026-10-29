"""CLI for the shared evaluator; validation is the default split."""

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

if __package__:
    from .data_utils import load_inputs, load_jsonl, read_csv
    from .evaluation import evaluate_rows
else:
    from project.common.data_utils import load_inputs, load_jsonl, read_csv
    from project.common.evaluation import evaluate_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--trace", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", choices=["validation", "test"], default="validation")
    parser.add_argument("--model-config", type=Path, default=Path(__file__).with_name("config.json"))
    args = parser.parse_args()
    if args.split == "test":
        config = json.loads(args.model_config.read_text(encoding="utf-8"))
        if config.get("final_test_frozen") is not True:
            raise ValueError("Freeze the final test configuration before opening final-test inputs or references")
    inputs = load_inputs(args.inputs, allow_test=args.split == "test")
    references = read_csv(args.references, ["Sample_id", "Answer"], allow_test=args.split == "test")
    predictions = read_csv(args.predictions, ["Sample_id", "Answer"])
    traces = load_jsonl(args.trace) if args.trace else None
    metrics, details = evaluate_rows(inputs, references, predictions, traces=traces)
    files = {"inputs": args.inputs, "references": args.references, "predictions": args.predictions}
    if args.trace:
        files["trace"] = args.trace
    metrics["split"] = args.split
    metrics["input_sha256"] = {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in files.items()}
    common_dir = Path(__file__).resolve().parent
    metrics["scoring_sha256"] = {
        name: hashlib.sha256((common_dir / name).read_bytes()).hexdigest()
        for name in ["evaluate.py", "evaluation.py", "evaluation_config.json", "data_utils.py"]
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    with (args.output_dir / "per_sample.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(details[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(details)
    print(json.dumps(metrics, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
