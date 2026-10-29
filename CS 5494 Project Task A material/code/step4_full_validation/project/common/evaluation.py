"""Reference-blind answer extraction and deterministic shared metrics.

No torch/model dependency. Text normalization deliberately preserves punctuation
and articles because this course includes short Python questions.
"""

from collections import Counter
from decimal import Decimal, InvalidOperation
from fractions import Fraction
import json
import re
import unicodedata

from .data_utils import TASKS, index_by_id, require_same_ids

VERSION = "evaluation_v2"
ABS_TOL = Fraction(1, 1_000_000)
REL_TOL = Fraction(0)
NUMBER = re.compile(r"[+-]?(?:(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\Z")
GROUPED = re.compile(r"[+-]?\d{1,3}(?:,\d{3})+(?:\.\d+)?(?:[eE][+-]?\d+)?\Z")
FINAL_MARKER = re.compile(r"(?:^|\n)\s*(?:final\s+(?:numeric\s+answer(?:\s+only)?|answer)|answer|最终答案|答案)\s*[:：]\s*", re.I)


def normalize_text(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def strict_text(text):
    return " ".join(unicodedata.normalize("NFKC", text).split())


def tokens(text):
    return re.findall(r"\w+|[^\w\s]", normalize_text(text), flags=re.UNICODE)


def token_f1(prediction, reference):
    pred, gold = tokens(prediction), tokens(reference)
    if not pred or not gold:
        return float(pred == gold)
    overlap = sum((Counter(pred) & Counter(gold)).values())
    return 2 * overlap / (len(pred) + len(gold))


def remove_math_wrappers(text):
    value = unicodedata.normalize("NFKC", text).strip().replace("−", "-")
    for _ in range(5):
        original = value
        fence = re.fullmatch(r"```(?:text|plaintext|math|latex)?\s*\n?(.*?)\n?```", value, re.S | re.I)
        if fence:
            value = fence.group(1).strip()
        for left, right in [("**", "**"), ("$", "$"), ("\\(", "\\)"), ("\\[", "\\]")]:
            if value.startswith(left) and value.endswith(right) and len(value) > len(left) + len(right):
                value = value[len(left):-len(right)].strip()
        boxed = re.fullmatch(r"\\boxed\{(.*)\}", value, re.S)
        if boxed:
            value = boxed.group(1).strip()
        if value == original:
            break
    return value


def decimal_fraction(value):
    if len(value) > 256:
        raise ValueError("Numeric literal too long")
    if "," in value:
        if not GROUPED.fullmatch(value):
            raise ValueError("Invalid thousands grouping")
        value = value.replace(",", "")
    if not NUMBER.fullmatch(value):
        raise ValueError("Not a complete number")
    decimal = Decimal(value)
    if not decimal.is_finite() or abs(decimal.as_tuple().exponent) > 1000:
        raise ValueError("Unsupported non-finite or extreme numeric value")
    return Fraction(decimal)


def parse_number(text):
    """Return an exact rational, or None; never evaluate arithmetic expressions."""
    value = remove_math_wrappers(text)
    if value.endswith("%"):
        value = value[:-1].strip()  # Dataset percentage answers use percentage points.
    tex_fraction = re.fullmatch(r"\\(?:d?frac)\{\s*([+-]?\d+)\s*\}\{\s*([+-]?\d+)\s*\}", value)
    if tex_fraction:
        value = "/".join(tex_fraction.groups())
    try:
        if "/" in value:
            if value.count("/") != 1:
                return None
            numerator, denominator = value.split("/")
            denominator = decimal_fraction(denominator.strip())
            return decimal_fraction(numerator.strip()) / denominator if denominator else None
        return decimal_fraction(value)
    except (ValueError, ZeroDivisionError, InvalidOperation, OverflowError):
        return None


def extract_answer(task, raw_answer, truncated=False):
    """No reference argument exists: extraction cannot inspect a gold answer."""
    if task not in TASKS or not isinstance(raw_answer, str):
        raise ValueError("Invalid task or raw answer")
    if truncated:
        return {"answer": "", "status": "truncated"}
    value = raw_answer.strip()
    if not value:
        return {"answer": "", "status": "blank"}
    if task != "math":
        return {"answer": value, "status": "text"}
    candidate = remove_math_wrappers(value)
    if parse_number(candidate) is not None:
        return {"answer": candidate, "status": "complete_number"}
    markers = list(FINAL_MARKER.finditer(value))
    if markers:
        candidate = remove_math_wrappers(value[markers[-1].end():])
        if parse_number(candidate) is not None:
            return {"answer": candidate, "status": "explicit_final_number"}
    return {"answer": "", "status": "unparseable"}


def reference_forms(row):
    canonical = row.get("Answer")
    if not isinstance(canonical, str) or not canonical.strip():
        raise ValueError(f"Empty reference: {row.get('Sample_id')}")
    alternatives = row.get("Accepted_answers", "")
    if isinstance(alternatives, str):
        try:
            alternatives = json.loads(alternatives) if alternatives.strip() else []
        except ValueError as exc:
            raise ValueError(f"Invalid Accepted_answers: {row.get('Sample_id')}") from exc
    if not isinstance(alternatives, list) or any(not isinstance(a, str) or not a.strip() for a in alternatives):
        raise ValueError(f"Invalid reference alternatives: {row.get('Sample_id')}")
    return list(dict.fromkeys([canonical] + alternatives))


def numeric_equal(left, right, abs_tol=ABS_TOL, rel_tol=REL_TOL):
    return abs(left - right) <= max(abs_tol, rel_tol * max(abs(left), abs(right)))


def score_answer(task, prediction, reference_row, truncated=False):
    extracted = extract_answer(task, prediction, truncated=truncated)
    answer, forms = extracted["answer"], reference_forms(reference_row)
    result = {"answer": answer, "extraction_status": extracted["status"],
              "exact_match": None, "token_f1": None, "strict_exact_match": None,
              "numeric_correct": None, "reference_unanswerable": None}
    if task == "math":
        gold_numbers = [parse_number(form) for form in forms]
        if any(number is None for number in gold_numbers):
            raise ValueError(f"Unparseable numeric reference: {reference_row.get('Sample_id')}")
        number = parse_number(answer)
        result["numeric_correct"] = float(number is not None and any(numeric_equal(number, gold) for gold in gold_numbers))
        return result
    if task == "rag" and normalize_text(reference_row["Answer"]) == "unanswerable":
        correct = float(normalize_text(answer) == "unanswerable")
        result.update(exact_match=correct, token_f1=correct,
                      strict_exact_match=float(strict_text(answer) == "UNANSWERABLE"), reference_unanswerable=True)
        return result
    result["reference_unanswerable"] = False if task == "rag" else None
    if not answer:
        result.update(exact_match=0.0, token_f1=0.0, strict_exact_match=0.0)
    else:
        result.update(
            exact_match=float(any(normalize_text(answer) == normalize_text(form) for form in forms)),
            token_f1=max(token_f1(answer, form) for form in forms),
            strict_exact_match=float(any(strict_text(answer) == strict_text(form) for form in forms)),
        )
    return result


def evaluate_rows(inputs, references, predictions, traces=None):
    input_by_id = index_by_id(inputs, "inputs")
    refs = index_by_id(references, "references")
    preds = index_by_id(predictions, "predictions")
    if not inputs:
        raise ValueError("Cannot evaluate an empty input set")
    require_same_ids(input_by_id, refs, "Reference")
    require_same_ids(input_by_id, preds, "Prediction")
    trace_by_id = index_by_id(traces, "trace") if traces is not None else None
    if trace_by_id is not None:
        require_same_ids(input_by_id, trace_by_id, "Trace")
    details = []
    for row in inputs:
        sample_id, task = row["Sample_id"], row["Task"]
        if task not in TASKS:
            raise ValueError(f"Unknown task: {task}")
        if refs[sample_id].get("Task", task) != task:
            raise ValueError(f"Reference task differs: {sample_id}")
        prediction = preds[sample_id].get("Answer")
        if not isinstance(prediction, str):
            raise ValueError(f"Prediction must contain an Answer string: {sample_id}")
        trace = trace_by_id[sample_id] if trace_by_id is not None else None
        raw, truncated = prediction, False
        if trace is not None:
            if trace.get("Task", task) != task:
                raise ValueError(f"Trace task differs: {sample_id}")
            raw, truncated = trace.get("raw_answer"), trace.get("truncated", False)
            if not isinstance(raw, str) or type(truncated) is not bool:
                raise ValueError(f"Trace requires a raw_answer string and boolean truncated flag: {sample_id}")
        score = score_answer(task, raw, refs[sample_id], truncated=truncated)
        if trace is not None and score["answer"] != prediction:
            raise ValueError(f"Trace extraction differs from prediction CSV: {sample_id}")
        details.append({"Sample_id": sample_id, "Task": task, "raw_answer": raw, **score})

    def percent(values):
        return 100 * sum(values) / len(values) if values else None

    per_task = {}
    for task in TASKS:
        task_rows = [row for row in details if row["Task"] == task]
        result = {"count": len(task_rows), "empty_extracted_answers": sum(not row["answer"] for row in task_rows),
                  "extraction_status_counts": dict(Counter(row["extraction_status"] for row in task_rows))}
        if task == "math":
            result["numeric_accuracy"] = percent([row["numeric_correct"] for row in task_rows])
        else:
            for key in ["exact_match", "token_f1", "strict_exact_match"]:
                result[key] = percent([row[key] for row in task_rows])
            if task == "rag":
                no_answer_rows = [row for row in task_rows if row["reference_unanswerable"]]
                result["unanswerable_count"] = len(no_answer_rows)
                result["unanswerable_accuracy"] = percent([row["exact_match"] for row in no_answer_rows])
        per_task[task] = result
    primary = [per_task["qa"]["exact_match"], per_task["rag"]["exact_match"], per_task["math"]["numeric_accuracy"]]
    metrics = {"evaluation_version": VERSION, "count": len(details), "score_unit": "percent",
               "per_task": per_task, "macro_primary_score": sum(primary) / 3 if all(value is not None for value in primary) else None,
               "all_expected_ids_present": True, "absolute_tolerance": "1e-6", "relative_tolerance": "0"}
    return metrics, details
