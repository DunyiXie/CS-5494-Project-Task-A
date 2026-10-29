"""Actual 907-row validation baseline; run with %run -i in the GPU Colab."""

import argparse
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]


def main(namespace=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path("/content"))
    parser.add_argument("--no-download", action="store_true")
    args = parser.parse_args()
    namespace = namespace if namespace is not None else {}
    for name in list(sys.modules):
        if name == "project" or name.startswith("project.common") or name.startswith("project.step4_baseline_validation"):
            del sys.modules[name]
    sys.path.insert(0, str(ROOT))
    from project.common.data_utils import index_by_id, load_jsonl, read_csv, require_same_ids
    from project.common.inference import TorchBackend, read_inference_inputs, reusable_model_reason, run_inference, save_json, sha256
    from project.step4_baseline_validation.runner_helpers import (
        REFERENCE_RELATIVE, install_reference_guard, recheck_prompts, validate_backend_environment, validate_preflight, verify_payload)
    manifest = verify_payload(ROOT)
    data_dir = ROOT / "project/data/validation"
    config_path = ROOT / "project/common/config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    selection = json.loads((data_dir / "full_validation_manifest.json").read_text(encoding="utf-8"))
    if selection["count"] != 907 or selection["task_counts"] != {"qa": 300, "rag": 320, "math": 287}:
        raise ValueError("Expected the complete fixed validation subset")
    inputs_path = data_dir / "validation_inputs.csv"
    inputs = read_inference_inputs(inputs_path, config)
    profile_path = data_dir / "passed_preflight_summary.json"
    length_path = data_dir / "passed_preflight_lengths.csv"
    if sha256(profile_path) != selection["preflight_summary_sha256"] or sha256(length_path) != selection["preflight_lengths_sha256"]:
        raise ValueError("Passed preflight artifacts changed")
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    lengths = read_csv(length_path)
    config, recorded = validate_preflight(inputs, config_path, profile, lengths, selection, inputs_path)
    run_name = "step4_4_baseline_validation_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f_UTC")
    run_dir = args.output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    for relative in manifest["sha256"]:
        if relative == REFERENCE_RELATIVE:
            continue
        target = run_dir / "reproduce" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, target)
    shutil.copyfile(ROOT / "manifest.json", run_dir / "reproduce/manifest.json")
    result = {"status": "running", "scope": "All 907 validation examples; no final test inference",
              "count": len(inputs), "task_counts": selection["task_counts"],
              "gpu_execution": "pending", "test_data_read": False,
              "final_test_frozen": config["final_test_frozen"],
              "bundle_reference_verified_after_generation": False}
    save_json(run_dir / "runner_summary.json", result)
    reference_path = ROOT / REFERENCE_RELATIVE
    guard = install_reference_guard(reference_path)
    try:
        try:
            import torch
            if not torch.cuda.is_available():
                raise RuntimeError("Enable the Colab GPU for the full validation baseline")
            print("GPU:", torch.cuda.get_device_name(0), flush=True)
            previous_model = namespace.get("fp16_model")
            if previous_model is None:
                previous_model = namespace.get("model")
            previous_tokenizer = namespace.get("tokenizer")
            reason = reusable_model_reason(previous_model, previous_tokenizer, config)
            if reason is not None:
                print("Loading the fixed FP16 model; reuse check:", reason, flush=True)
                if previous_model is not None:
                    for key in [key for key in namespace if namespace[key] is previous_model]:
                        namespace[key] = None
                previous_model = None
                previous_tokenizer = None
                gc.collect()
                torch.cuda.empty_cache()
            else:
                print("Reusing the verified FP16 model and tokenizer.", flush=True)
            backend = TorchBackend(config, previous_model, previous_tokenizer)
            namespace["model"] = namespace["fp16_model"] = backend.model
            namespace["tokenizer"] = backend.tokenizer
            observed = json.loads((ROOT / "project/common/environment_observed.json").read_text(encoding="utf-8"))
            save_json(run_dir / "backend_environment.json", backend.metadata)
            result["environment_checks"] = validate_backend_environment(backend.metadata, observed, config)
            result["gpu_execution"] = "model_loaded"
            print("Rechecking all 907 prompts against the passed tokenizer preflight.", flush=True)
            recheck = recheck_prompts(inputs, backend.tokenizer, config, recorded)
            if recheck["count"] != 907:
                raise ValueError("Not all validation prompts were rechecked")
            save_json(run_dir / "preflight_recheck.json", recheck)
            result["prompt_parity_verified_count"] = recheck["count"]
            result["gpu_execution"] = "generation_started"
            save_json(run_dir / "runner_summary.json", result)
            summary = run_inference(inputs, backend, config, run_dir / "inference", method="baseline")
            result.update(gpu_execution="completed", model_reused=backend.metadata["model_reused"],
                          reference_reads_during_inference=guard["blocked_attempts"])
        finally:
            guard["active"] = False
        verify_payload(ROOT, include_references=True)
        if sha256(reference_path) != selection["reference_sha256"]:
            raise ValueError("Validation reference file changed")
        result["bundle_reference_verified_after_generation"] = True
        target = run_dir / "reproduce" / REFERENCE_RELATIVE
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(reference_path, target)
        command = [sys.executable, str(ROOT / "project/common/evaluate.py"),
                   "--inputs", str(inputs_path), "--references", str(reference_path),
                   "--predictions", str(run_dir / "inference/predictions.csv"),
                   "--trace", str(run_dir / "inference/raw_outputs.jsonl"),
                   "--output-dir", str(run_dir / "evaluation")]
        process = subprocess.run(command, cwd=str(ROOT), capture_output=True, text=True, encoding="utf-8")
        if process.returncode:
            raise RuntimeError(process.stderr)
        (run_dir / "evaluation/console_output.txt").write_text(process.stdout, encoding="utf-8")
        predictions = read_csv(run_dir / "inference/predictions.csv", ["Sample_id", "Answer"])
        traces = load_jsonl(run_dir / "inference/raw_outputs.jsonl")
        require_same_ids(index_by_id(inputs), index_by_id(predictions), "Predictions")
        require_same_ids(index_by_id(inputs), index_by_id(traces), "Traces")
        metrics = json.loads((run_dir / "evaluation/metrics.json").read_text(encoding="utf-8"))
        if (len(predictions) != 907 or len(traces) != 907 or metrics["count"] != 907
                or summary["status"] != "complete" or guard["blocked_attempts"] != 0
                or [metrics["per_task"][task]["count"] for task in ["qa", "rag", "math"]] != [300, 320, 287]):
            raise ValueError("Full validation inference or evaluation is incomplete")
        result.update(status="STEP4_4_FULL_VALIDATION_BASELINE_PASSED",
                      completed_count=summary["completed_count"], blank_raw_answers=summary["blank_raw_answers"],
                      empty_extracted_answers=summary["empty_extracted_answers"], truncated_outputs=summary["truncated_outputs"],
                      max_input_tokens=summary["max_input_tokens"], summed_generation_seconds=summary["summed_generation_seconds"],
                      generation_wall_seconds=summary["generation_wall_seconds"], gpu_memory=summary["gpu_memory"],
                      per_task=metrics["per_task"], macro_primary_score=metrics["macro_primary_score"],
                      interpretation="Complete validation baseline; incorrect/empty/truncated answers remain in the denominator")
        save_json(run_dir / "runner_summary.json", result)
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    except BaseException as exc:
        guard["active"] = False
        result.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        save_json(run_dir / "runner_summary.json", result)
        raise
    finally:
        archive = shutil.make_archive(str(run_dir), "zip", root_dir=run_dir)
        namespace["STEP4_4_VALIDATION_RESULT_ZIP"] = archive
        print("Result ZIP:", archive, flush=True)
        if not args.no_download:
            try:
                from google.colab import files
                files.download(archive)
            except Exception as exc:
                print("Result ZIP saved; retry files.download(STEP4_4_VALIDATION_RESULT_ZIP):", str(exc), flush=True)


if __name__ == "__main__":
    main(globals())
