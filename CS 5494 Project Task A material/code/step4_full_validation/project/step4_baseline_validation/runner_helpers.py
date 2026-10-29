"""Reference-blind gates for the complete validation baseline runner."""

from collections import Counter
import json
import os
from pathlib import Path
import re
import sys

from project.common.data_utils import index_by_id, require_same_ids
from project.common.inference import prepare_prompt, sha256, validate_config

REFERENCE_RELATIVE = "project/data/validation/validation_reference.csv"
CRITICAL_PACKAGES = ("transformers", "accelerate", "huggingface-hub", "tokenizers", "safetensors")


def verify_payload(root, include_references=False):
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    for relative, expected in manifest["sha256"].items():
        if relative == REFERENCE_RELATIVE and not include_references:
            continue
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or not path.is_file() or sha256(path) != expected:
            raise ValueError(f"Bundle file differs: {relative}")
    return manifest


def install_reference_guard(reference_path):
    state = {"active": True, "blocked_attempts": 0}
    forbidden = reference_path.resolve()
    def guard(event, args):
        if state["active"] and event == "open" and isinstance(args[0], (str, bytes, os.PathLike)):
            path = Path(os.fsdecode(args[0]))
            if path.resolve() == forbidden or path.name.casefold() == "test.csv" or path.name.casefold().startswith("test_answer"):
                state["blocked_attempts"] += 1
                raise RuntimeError("Reference/test reads are blocked during model inference")
    sys.addaudithook(guard)
    return state


def validate_preflight(inputs, config_path, profile, lengths, selection, input_path):
    config = validate_config(json.loads(Path(config_path).read_text(encoding="utf-8")))
    expected_count = selection["count"]
    input_by_id = index_by_id(inputs, "validation inputs")
    length_by_id = index_by_id(lengths, "preflight lengths")
    require_same_ids(input_by_id, length_by_id, "Length preflight")
    if (len(inputs) != expected_count or [row["Sample_id"] for row in inputs] != selection["sample_ids"]
            or [row["Sample_id"] for row in lengths] != selection["sample_ids"]
            or dict(Counter(row["Task"] for row in inputs)) != selection["task_counts"]):
        raise ValueError("Validation IDs, order or task counts differ from the frozen subset")
    if (profile["status"] != "STEP4_3_LENGTH_PREFLIGHT_PASSED" or profile["ready_for_generation"] is not True
            or profile["count"] != expected_count or profile["over_budget_count"] != 0 or profile["over_budget_ids"]
            or profile["inference_parity_verified_count"] != expected_count
            or profile["smoke_parity"]["matched"] != 10 or profile["smoke_parity"]["expected"] != 10):
        raise ValueError("Actual-tokenizer preflight is incomplete or not ready")
    if (sha256(input_path) != selection["input_sha256"] or sha256(input_path) != profile["input_sha256"]
            or sha256(config_path) != selection["config_sha256"] or sha256(config_path) != profile["config_sha256"]):
        raise ValueError("Inputs or raw config file changed after the passed preflight")
    limit = config["input_processing"]["max_input_tokens"]
    output = config["generation"]["max_new_tokens"]
    normalized = {}
    for row in lengths:
        tokens = int(row["input_tokens"])
        sample_id = row["Sample_id"]
        if (row["Task"] != input_by_id[sample_id]["Task"] or not 0 < tokens <= limit
                or row["over_budget"] not in (False, "False")
                or int(row["input_budget"]) != limit or int(row["output_budget"]) != output
                or int(row["total_reserved_tokens"]) != tokens + output
                or int(row["remaining_input_tokens"]) != limit - tokens
                or not re.fullmatch(r"[0-9a-f]{64}", row["prompt_sha256"])):
            raise ValueError(f"Invalid or over-budget input length record: {sample_id}")
        normalized[sample_id] = {"input_tokens": tokens, "prompt_sha256": row["prompt_sha256"], "Task": row["Task"]}
    if profile["overall"]["max"] != max(row["input_tokens"] for row in normalized.values()):
        raise ValueError("Length profile maximum differs from per-example records")
    return config, normalized


def recheck_prompts(inputs, tokenizer, config, recorded):
    checked = []
    for row in inputs:
        prepared = prepare_prompt(row, tokenizer, config)
        previous = recorded[row["Sample_id"]]
        if prepared["input_tokens"] != previous["input_tokens"] or prepared["prompt_sha256"] != previous["prompt_sha256"]:
            raise ValueError(f"Current tokenizer/prompt differs from actual preflight: {row['Sample_id']}")
        checked.append({"Sample_id": row["Sample_id"], "input_tokens": prepared["input_tokens"],
                        "prompt_sha256": prepared["prompt_sha256"]})
    return {"status": "passed", "count": len(checked), "max_input_tokens": max(row["input_tokens"] for row in checked),
            "checks": checked}


def validate_backend_environment(actual, observed, config):
    if (actual["backend_kind"] != "torch_cuda" or actual["adapter"] is not None or actual["quantized"] is not False
            or actual["dtype"] != "float16" or actual["floating_parameter_dtypes"] != ["torch.float16"]
            or actual["parameter_devices"] != ["cuda:0"]
            or actual["model_id"] != config["model_id"]
            or actual["model_revision"] != config["model_revision"] or actual["tokenizer_revision"] != config["tokenizer_revision"]
            or actual["attention_implementation"] != config["attention_implementation"]):
        raise ValueError("Full baseline requires the pinned, unquantized FP16 CUDA model without an adapter")
    differences = {name: {"expected": observed["packages"].get(name), "actual": actual["packages"].get(name)}
                   for name in CRITICAL_PACKAGES if actual["packages"].get(name) != observed["packages"].get(name)}
    for name in ("torch", "cuda"):
        if actual[name] != observed[name]:
            differences[name] = {"expected": observed[name], "actual": actual[name]}
    if differences:
        raise ValueError(f"Inference dependencies changed from the verified environment: {differences}")
    if actual["snapshot_config_sha256"] != observed["snapshot_config_sha256"]:
        raise ValueError("Model/tokenizer snapshot files differ from the verified environment")
    generation = actual["generation_effective"]
    if (generation["do_sample"] is not False or generation["num_beams"] != 1
            or generation["max_new_tokens"] != config["generation"]["max_new_tokens"]
            or generation["eos_token_id"] != config["generation"]["eos_token_id"]
            or generation["pad_token_id"] != config["generation"]["pad_token_id"]):
        raise ValueError("Effective decoding differs from the validated configuration")
    return {"status": "passed", "critical_dependencies_match": True, "fixed_snapshot_matches": True,
            "fp16_cuda_baseline_verified": True}
