"""Stage 1: generate all frozen Task A predictions, with no reference file present."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
import shutil
import statistics
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from runner_helpers import (EXPECTED_COUNT, FREEZE_ID, FREEZE_SHA256, INFERENCE_COMPLETE,
                            check_generation_complete, install_reference_guard, load_selected_inputs,
                            read_json, save_json, sha256, validate_backend, verify_bundle, verify_freeze)


def main(namespace=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path("/content"))
    parser.add_argument("--no-download", action="store_true")
    args = parser.parse_args()
    namespace = namespace if namespace is not None else {}
    run_name = "task_A_test_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f_UTC")
    run_dir = args.output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    namespace["TASK_A_TEST_RUN_DIR"] = str(run_dir)
    result = {"status": "running", "scope": "All 1007 final-test inputs; A frozen baseline",
              "freeze_id": FREEZE_ID, "count": EXPECTED_COUNT, "completed_count": 0,
              "final_test_frozen": True, "gpu_execution": "pending", "reference_answers_read": False,
              "reference_reads_during_inference": 0}
    save_json(run_dir / "runner_summary.json", result)
    guard = install_reference_guard()
    try:
        manifest = verify_bundle(ROOT)
        freeze = verify_freeze(ROOT, check_runtime=True)
        save_json(run_dir / "freeze_verification.json", freeze)
        from project.common.inference import TorchBackend, prepare_prompt, run_inference
        from project.common.data_utils import load_jsonl
        config = read_json(ROOT / "frozen/config_frozen.json")
        inputs, selection = load_selected_inputs(ROOT, config)
        result["task_counts"] = selection["task_counts"]
        shutil.copyfile(ROOT / "input_manifest.json", run_dir / "input_manifest.json")
        for relative in manifest["sha256"]:
            target = run_dir / "reproduce" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / relative, target)
        shutil.copyfile(ROOT / "bundle_manifest.json", run_dir / "reproduce/bundle_manifest.json")
        import torch
        print("GPU:", torch.cuda.get_device_name(0), flush=True)
        print("Loading a fresh pinned FP16 model for Task A; no prior notebook model is reused.", flush=True)
        gc.collect()
        torch.cuda.empty_cache()
        backend = TorchBackend(config)
        save_json(run_dir / "backend_environment.json", backend.metadata)
        result["environment_checks"] = validate_backend(backend.metadata,
                                                         read_json(ROOT / "frozen/environment_lock.json"), config)
        print("Checking token lengths for all 1007 final-test inputs.", flush=True)
        lengths = []
        for row in inputs:
            prompt = prepare_prompt(row, backend.tokenizer, config)
            lengths.append({"Sample_id": row["Sample_id"], "Task": row["Task"],
                            "input_tokens": prompt["input_tokens"], "prompt_sha256": prompt["prompt_sha256"]})
        import csv
        with (run_dir / "lengths.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(lengths[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(lengths)
        counts = [r["input_tokens"] for r in lengths]
        profile = {"status": "TASK_A_TEST_LENGTH_PREFLIGHT_PASSED", "count": len(lengths),
                   "ready_for_generation": True, "over_budget_count": 0, "target_truncation": False,
                   "reference_answers_read": False, "input_budget": config["input_processing"]["max_input_tokens"],
                   "output_budget": config["generation"]["max_new_tokens"],
                   "min": min(counts), "median": statistics.median(counts),
                   "mean": statistics.mean(counts), "max": max(counts),
                   "per_task_max": {t: max(r["input_tokens"] for r in lengths if r["Task"] == t)
                                    for t in selection["task_counts"]}}
        save_json(run_dir / "length_preflight.json", profile)
        print(json.dumps(profile, ensure_ascii=False, indent=2), flush=True)
        result["gpu_execution"] = "generation_started"
        save_json(run_dir / "runner_summary.json", result)
        summary = run_inference(inputs, backend, config, run_dir / "inference", method="baseline")
        prompts = load_jsonl(run_dir / "inference/prompts.jsonl")
        if any((p["Sample_id"], p["input_tokens"], p["prompt_sha256"]) !=
               (r["Sample_id"], r["input_tokens"], r["prompt_sha256"]) for p, r in zip(prompts, lengths)):
            raise ValueError("Actual inference prompts differ from length preflight")
        if len(prompts) != EXPECTED_COUNT or summary["completed_count"] != EXPECTED_COUNT or guard["blocked_attempts"]:
            raise ValueError("Incomplete generation or a forbidden reference-read attempt")
        artifacts = ["backend_environment.json", "freeze_verification.json", "input_manifest.json",
                     "length_preflight.json", "lengths.csv", "inference/inputs.csv", "inference/config_snapshot.json",
                     "inference/environment.json", "inference/prompts.jsonl", "inference/raw_outputs.jsonl",
                     "inference/predictions.csv", "inference/run_summary.json"]
        receipt = {"freeze_id": FREEZE_ID, "freeze_manifest_sha256": FREEZE_SHA256,
                   "bundle_manifest_sha256": sha256(ROOT / "bundle_manifest.json"),
                   "input_sha256": selection["input_sha256"], "method": "baseline",
                   "sha256": {p: sha256(run_dir / p) for p in artifacts}}
        save_json(run_dir / "generation_receipt.json", receipt)
        result.update(status=INFERENCE_COMPLETE, gpu_execution="completed", completed_count=summary["completed_count"],
                      reference_reads_during_inference=guard["blocked_attempts"],
                      generation_receipt_sha256=sha256(run_dir / "generation_receipt.json"),
                      blank_raw_answers=summary["blank_raw_answers"], empty_extracted_answers=summary["empty_extracted_answers"],
                      truncated_outputs=summary["truncated_outputs"], max_input_tokens=summary["max_input_tokens"],
                      summed_generation_seconds=summary["summed_generation_seconds"],
                      generation_wall_seconds=summary["generation_wall_seconds"], gpu_memory=summary["gpu_memory"])
        save_json(run_dir / "runner_summary.json", result)
        check_generation_complete(ROOT, run_dir)
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
        print("Predictions are complete and preserved. Now upload test_answer.csv in the separate scoring cell.", flush=True)
    except BaseException as exc:
        result.update(status="failed", error_type=type(exc).__name__, error=str(exc),
                      reference_reads_during_inference=guard["blocked_attempts"])
        save_json(run_dir / "runner_summary.json", result)
        raise
    finally:
        guard["active"] = False
        archive = shutil.make_archive(str(run_dir) + "_predictions", "zip", root_dir=run_dir)
        namespace["TASK_A_TEST_PREDICTION_ZIP"] = archive
        print("Prediction ZIP:", archive, flush=True)
        if not args.no_download:
            try:
                from google.colab import files
                files.download(archive)
            except Exception as exc:
                print("ZIP saved. Retry files.download(TASK_A_TEST_PREDICTION_ZIP):", str(exc), flush=True)


if __name__ == "__main__":
    main(globals())
