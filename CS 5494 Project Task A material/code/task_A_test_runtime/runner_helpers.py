"""Gates for the A-only, frozen, two-stage final test. No reference is read here."""

from collections import Counter
import hashlib
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import sys

FREEZE_SHA256 = "973123c9c7ef06780a2dfc9f3f77d2441d4ddd2fddc2d535296296aeca32c7f8"
FREEZE_ID = "TaskA_baseline_fp16_evaluation_v2_20261027_v1"
EXPECTED_COUNT = 1007
INFERENCE_COMPLETE = "TASK_A_TEST_INFERENCE_COMPLETE_PENDING_SCORING"
INPUT_FIELDS = ["Sample_id", "Task", "Question", "Context"]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_bytes((json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8"))


def safe_path(root, name):
    rel = PurePosixPath(name)
    if rel.is_absolute() or ".." in rel.parts or "\\" in name or ":" in name:
        raise ValueError("Unsafe relative path: " + name)
    path = Path(root).joinpath(*rel.parts)
    if not path.resolve().is_relative_to(Path(root).resolve()) or path.is_symlink():
        raise ValueError("Path is outside the runtime: " + name)
    return path


def verify_freeze(root, check_runtime=False):
    frozen = Path(root) / "frozen"
    if sha256(frozen / "freeze_manifest.json") != FREEZE_SHA256:
        raise ValueError("The original A freeze manifest SHA does not match")
    for name, entry in read_json(frozen / "freeze_manifest.json")["files"].items():
        path = safe_path(frozen, name)
        if path.stat().st_size != entry["bytes"] or sha256(path) != entry["sha256"]:
            raise ValueError("Frozen source changed before import: " + name)
    spec = importlib.util.spec_from_file_location("task_a_frozen_verifier", frozen / "verify_freeze.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    result = module.verify(frozen, check_runtime)
    if result["freeze_id"] != FREEZE_ID:
        raise ValueError("Unexpected A freeze ID")
    return result


def verify_bundle(root):
    root = Path(root).resolve()
    manifest = read_json(root / "bundle_manifest.json")
    if (manifest["freeze_manifest_sha256"] != FREEZE_SHA256 or manifest["freeze_id"] != FREEZE_ID
            or manifest["references_included"] is not False or manifest["method"] != "baseline"):
        raise ValueError("Unexpected test runtime manifest")
    for name, wanted in manifest["sha256"].items():
        if PurePosixPath(name).name.casefold().startswith("test_answer"):
            raise ValueError("Test references must not be packaged with inference")
        if sha256(safe_path(root, name)) != wanted:
            raise ValueError("Runtime file changed: " + name)
    return manifest


def install_reference_guard():
    state = {"active": True, "blocked_attempts": 0}

    def guard(event, args):
        if state["active"] and event == "open" and isinstance(args[0], (str, bytes, os.PathLike)):
            path = Path(os.fsdecode(args[0]))
            if path.name.casefold().startswith("test_answer"):
                state["blocked_attempts"] += 1
                raise RuntimeError("Test reference reads are blocked until generation is complete")
    sys.addaudithook(guard)
    return state


def load_selected_inputs(root, config):
    from project.common.inference import read_inference_inputs
    import csv
    root = Path(root)
    path = root / "inputs/test.csv"
    selection = read_json(root / "input_manifest.json")
    if selection["freeze_id"] != FREEZE_ID or selection["freeze_manifest_sha256"] != FREEZE_SHA256:
        raise ValueError("Inputs were not prepared for the original A freeze")
    if sha256(path) != selection["input_sha256"]:
        raise ValueError("Test inputs changed after preparation")
    with path.open(encoding="utf-8-sig", newline="") as stream:
        if next(csv.reader(stream)) != INPUT_FIELDS:
            raise ValueError("Test inputs must contain exactly four input columns")
    rows = read_inference_inputs(path, config, split="test")
    if (len(rows) != EXPECTED_COUNT or selection["count"] != EXPECTED_COUNT
            or [row["Sample_id"] for row in rows] != selection["sample_ids"]
            or dict(Counter(row["Task"] for row in rows)) != selection["task_counts"]):
        raise ValueError("Test IDs, order, count or task counts differ")
    return rows, selection


def validate_backend(actual, lock, config):
    expected = {"backend_kind": "torch_cuda", "model_id": config["model_id"],
                "model_revision": config["model_revision"], "tokenizer_revision": config["tokenizer_revision"],
                "dtype": "float16", "quantized": False, "adapter": None,
                "attention_implementation": "sdpa", "parameter_devices": ["cuda:0"],
                "floating_parameter_dtypes": ["torch.float16"], "model_reused": False,
                "torch": lock["torch"], "cuda": lock["cuda"], "gpu": lock["gpu"],
                "snapshot_config_sha256": lock["snapshot_config_sha256"],
                "generation_effective": lock["recorded_effective_generation"]}
    differences = {name: {"expected": wanted, "actual": actual.get(name)}
                   for name, wanted in expected.items() if actual.get(name) != wanted}
    differences.update({"package:" + name: {"expected": wanted, "actual": actual.get("packages", {}).get(name)}
                        for name, wanted in lock["critical_packages"].items()
                        if actual.get("packages", {}).get(name) != wanted})
    if differences:
        raise ValueError("Frozen model/environment mismatch: " + json.dumps(differences))
    return {"status": "passed", "fresh_pinned_fp16_model": True,
            "critical_dependencies_match": True, "snapshot_matches": True, "all_generation_parameters_match": True}


def check_generation_complete(root, run_dir, allow_simulated=False):
    """Fail before any reference file is opened. CLI never enables allow_simulated."""
    verify_freeze(root)
    from project.common.data_utils import load_jsonl, read_csv
    from project.common.evaluation import extract_answer
    from project.common.inference import generation_record, user_message
    root, run_dir = Path(root), Path(run_dir)
    manifest = verify_bundle(root)
    config = read_json(root / "frozen/config_frozen.json")
    rows, selection = load_selected_inputs(root, config)
    summary = read_json(run_dir / "runner_summary.json")
    required_summary = {"status": INFERENCE_COMPLETE, "freeze_id": FREEZE_ID,
                        "count": EXPECTED_COUNT, "completed_count": EXPECTED_COUNT,
                        "reference_answers_read": False, "reference_reads_during_inference": 0,
                        "final_test_frozen": True, "task_counts": selection["task_counts"]}
    if any(summary.get(k) != v for k, v in required_summary.items()):
        raise ValueError("Inference is not complete or references were accessed")
    if sha256(run_dir / "generation_receipt.json") != summary["generation_receipt_sha256"]:
        raise ValueError("Generation receipt changed")
    receipt = read_json(run_dir / "generation_receipt.json")
    if (receipt["freeze_manifest_sha256"] != FREEZE_SHA256 or receipt["freeze_id"] != FREEZE_ID
            or receipt["bundle_manifest_sha256"] != sha256(root / "bundle_manifest.json")
            or receipt["input_sha256"] != selection["input_sha256"] or receipt["method"] != "baseline"):
        raise ValueError("Generation provenance differs from the issued test runtime")
    required_files = {"backend_environment.json", "freeze_verification.json", "input_manifest.json",
                      "length_preflight.json", "lengths.csv", "inference/inputs.csv", "inference/config_snapshot.json",
                      "inference/environment.json", "inference/prompts.jsonl", "inference/raw_outputs.jsonl",
                      "inference/predictions.csv", "inference/run_summary.json"}
    if set(receipt["sha256"]) != required_files:
        raise ValueError("Generation receipt artifact list is incomplete")
    for name, wanted in receipt["sha256"].items():
        if sha256(safe_path(run_dir, name)) != wanted:
            raise ValueError("Generated artifact changed: " + name)
    if read_json(run_dir / "inference/config_snapshot.json") != config:
        raise ValueError("Actual inference config differs from A freeze")
    if read_json(run_dir / "input_manifest.json") != selection:
        raise ValueError("Actual input manifest differs")
    proof = read_json(run_dir / "freeze_verification.json")
    if (proof["status"] != "TASK_A_FROZEN_RELEASE_VERIFIED" or proof["freeze_id"] != FREEZE_ID
            or proof["manifest_sha256"] != FREEZE_SHA256
            or (not allow_simulated and proof["runtime"]["checked"] is not True)):
        raise ValueError("Frozen runtime verification was not completed")
    actual = read_json(run_dir / "backend_environment.json")
    if read_json(run_dir / "inference/environment.json") != actual:
        raise ValueError("Actual environment records differ")
    if not allow_simulated:
        validate_backend(actual, read_json(root / "frozen/environment_lock.json"), config)
        if summary.get("gpu_execution") != "completed":
            raise ValueError("Actual GPU inference has not completed")
    elif actual.get("backend_kind") != "simulated_cpu_test_only":
        raise ValueError("CPU test mode requires an explicitly simulated backend")
    inner = read_json(run_dir / "inference/run_summary.json")
    if (inner["status"] != "complete" or inner["method"] != "baseline"
            or inner["completed_count"] != EXPECTED_COUNT or inner["evaluation_version"] != "evaluation_v2"
            or inner["actual_n_counts"] != {"0": EXPECTED_COUNT} or inner["training_pool_sha256"] is not None):
        raise ValueError("Baseline trace is incomplete or contains demonstrations")
    if (inner["requested_count"] != EXPECTED_COUNT or inner["task_counts"] != selection["task_counts"]
            or inner["backend_kind"] != actual["backend_kind"]
            or inner["input_sha256"] != sha256(run_dir / "inference/inputs.csv")
            or inner["config_sha256"] != sha256(run_dir / "inference/config_snapshot.json")
            or inner["predictions_sha256"] != sha256(run_dir / "inference/predictions.csv")):
        raise ValueError("Inference summary provenance is inconsistent")
    for name, wanted in inner["code_sha256"].items():
        if name not in ["inference.py", "inference_policy.json", "evaluation.py", "evaluation_config.json", "data_utils.py"]:
            raise ValueError("Unexpected inference source hash")
        if sha256(root / "frozen/project/common" / name) != wanted:
            raise ValueError("Inference used different frozen sources: " + name)
    if len(inner["code_sha256"]) != 5:
        raise ValueError("Missing inference source hashes")
    inputs = read_csv(run_dir / "inference/inputs.csv", INPUT_FIELDS)
    predictions = read_csv(run_dir / "inference/predictions.csv", ["Sample_id", "Answer"])
    traces = load_jsonl(run_dir / "inference/raw_outputs.jsonl")
    prompts = load_jsonl(run_dir / "inference/prompts.jsonl")
    lengths = read_csv(run_dir / "lengths.csv", ["Sample_id", "Task", "input_tokens", "prompt_sha256"])
    ids = selection["sample_ids"]
    if any([row["Sample_id"] for row in group] != ids for group in [inputs, predictions, traces, prompts, lengths]):
        raise ValueError("Inputs, prompts, predictions or traces have missing, extra or out-of-order IDs")
    if inputs != rows:
        raise ValueError("Generated input content changed")
    profile = read_json(run_dir / "length_preflight.json")
    if profile["count"] != EXPECTED_COUNT or profile["over_budget_count"] != 0 or not profile["ready_for_generation"]:
        raise ValueError("Final-test length preflight was not passed")
    if (profile["status"] != "TASK_A_TEST_LENGTH_PREFLIGHT_PASSED"
            or profile["max"] != max(int(r["input_tokens"]) for r in lengths)
            or profile["input_budget"] != config["input_processing"]["max_input_tokens"]
            or profile["output_budget"] != config["generation"]["max_new_tokens"]
            or profile["target_truncation"] is not False or profile["reference_answers_read"] is not False):
        raise ValueError("Length preflight summary differs from actual lengths or frozen policy")
    for row, prediction, trace, prompt, length in zip(rows, predictions, traces, prompts, lengths):
        expected_messages = [{"role": "system", "content": config["system_prompts"][row["Task"]]},
                             {"role": "user", "content": user_message(row, config)}]
        if (trace["Task"] != row["Task"] or trace["actual_n"] != 0
                or trace["requested_demo_ids"] or trace["used_demo_ids"] or trace["dropped_demo_ids"]
                or prompt["actual_n"] != 0 or prompt["requested_demo_ids"] or prompt["used_demo_ids"]
                or prompt["dropped_demo_ids"] or prompt["messages"] != expected_messages or prompt["Task"] != row["Task"]
                or trace["prompt_sha256"] != prompt["prompt_sha256"]
                or length["Task"] != row["Task"] or length["prompt_sha256"] != prompt["prompt_sha256"]
                or int(length["input_tokens"]) != prompt["input_tokens"]
                or hashlib.sha256(prompt["prompt"].encode("utf-8")).hexdigest() != prompt["prompt_sha256"]
                or not 0 < trace["input_tokens"] == prompt["input_tokens"] <= config["input_processing"]["max_input_tokens"]):
            raise ValueError("Invalid baseline prompt/trace metadata: " + row["Sample_id"])
        recreated = generation_record(trace["output_token_ids"], trace["raw_answer"], trace["raw_new_tokens_text"],
                                      trace["input_tokens"], trace["generation_seconds"], config)
        if any(trace.get(k) != v for k, v in recreated.items()):
            raise ValueError("Invalid stopping/token metadata: " + row["Sample_id"])
        extracted = extract_answer(row["Task"], trace["raw_answer"], trace["truncated"])
        if (prediction["Answer"] != extracted["answer"] or trace["Answer"] != extracted["answer"]
                or trace["extraction_status"] != extracted["status"]):
            raise ValueError("Prediction is not reference-blind extraction: " + row["Sample_id"])
    return {"status": "passed", "count": EXPECTED_COUNT, "task_counts": selection["task_counts"],
            "freeze_id": FREEZE_ID, "reference_file_opened_by_gate": False}
