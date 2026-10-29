"""Strict, standard-library-only input loading and Sample_id alignment."""

import csv
import json
from pathlib import Path

TASKS = ("qa", "rag", "math")


def read_csv(path, required_columns=(), allow_test=False):
    path = Path(path)
    if not allow_test and (path.name.casefold() == "test.csv" or path.name.casefold().startswith("test_answer")):
        raise ValueError("Test files are reserved for explicitly selected final-test mode.")
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        columns = reader.fieldnames or []
        if len(columns) != len(set(columns)) or not set(required_columns).issubset(columns):
            raise ValueError(f"Missing or duplicate CSV columns in {path.name}: {columns}")
        rows = list(reader)
    if any(None in row or any(value is None for value in row.values()) for row in rows):
        raise ValueError(f"Malformed CSV rows in {path.name}")
    index_by_id(rows, path.name)
    return rows


def index_by_id(rows, label="rows"):
    result = {}
    for row in rows:
        sample_id = row.get("Sample_id")
        if not isinstance(sample_id, str) or not sample_id.strip() or sample_id != sample_id.strip():
            raise ValueError(f"Blank or noncanonical Sample_id in {label}")
        if sample_id in result:
            raise ValueError(f"Duplicate Sample_id {sample_id} in {label}")
        result[sample_id] = row
    return result


def require_same_ids(expected, actual, label):
    missing, extra = set(expected) - set(actual), set(actual) - set(expected)
    if missing or extra:
        raise ValueError(f"{label} IDs do not match inputs: missing={sorted(missing)[:5]}, extra={sorted(extra)[:5]}")


def load_inputs(path, allow_test=False):
    rows = read_csv(path, ["Sample_id", "Task", "Question", "Context"], allow_test=allow_test)
    if not rows:
        raise ValueError("Input dataset is empty")
    for row in rows:
        if row["Task"] not in TASKS or not row["Question"].strip():
            raise ValueError(f"Invalid task or empty question: {row['Sample_id']}")
        if row["Task"] == "rag" and not row["Context"].strip():
            raise ValueError(f"RAG context is empty: {row['Sample_id']}")
    return rows


def write_predictions(path, rows):
    rows = list(rows)
    index_by_id(rows, "predictions")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["Sample_id", "Answer"], lineterminator="\n", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def load_jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    index_by_id(rows, "trace")
    return rows
