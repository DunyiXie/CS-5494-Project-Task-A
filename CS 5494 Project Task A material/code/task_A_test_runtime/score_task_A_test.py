"""Stage 2: verify complete frozen predictions before opening uploaded references."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from runner_helpers import (EXPECTED_COUNT, FREEZE_ID, FREEZE_SHA256, check_generation_complete,
                            read_json, save_json, sha256, verify_bundle, verify_freeze)


def main(namespace=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--no-download", action="store_true")
    args = parser.parse_args()
    namespace = namespace if namespace is not None else {}
    run_dir = args.run_dir.resolve()
    if (run_dir / "final_summary.json").exists():
        raise FileExistsError("This final test is already scored; preserve the completed result")
    # No references have been inspected, hashed or copied before these gates pass.
    verify_bundle(ROOT)
    freeze = verify_freeze(ROOT)
    completion = check_generation_complete(ROOT, run_dir)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f_UTC")
    scoring_dir = run_dir / ("scoring_" + timestamp)
    scoring_dir.mkdir(exist_ok=False)
    save_json(scoring_dir / "pre_reference_gate.json", completion)
    result = {"status": "scoring_started", "freeze_id": FREEZE_ID, "count": EXPECTED_COUNT,
              "generation_completion_verified_before_reference_access": True, "reference_answers_read": False}
    try:
        # This is the first reference read in either stage; predictions are already complete on disk.
        target = scoring_dir / "test_answer.csv"
        shutil.copyfile(args.references, target)
        result["reference_answers_read"] = True
        result["reference_sha256"] = sha256(target)
        evaluator = ROOT / "frozen/project/common/evaluate.py"
        command = [sys.executable, str(evaluator), "--split", "test",
                   "--model-config", str(ROOT / "frozen/config_frozen.json"),
                   "--inputs", str(ROOT / "inputs/test.csv"), "--references", str(target),
                   "--predictions", str(run_dir / "inference/predictions.csv"),
                   "--trace", str(run_dir / "inference/raw_outputs.jsonl"),
                   "--output-dir", str(scoring_dir / "evaluation")]
        process = subprocess.run(command, cwd=ROOT / "frozen", capture_output=True, text=True, encoding="utf-8")
        (scoring_dir / "console_output.txt").write_text(process.stdout + process.stderr, encoding="utf-8")
        if process.returncode:
            raise RuntimeError(process.stderr)
        metrics = read_json(scoring_dir / "evaluation/metrics.json")
        if (metrics["count"] != EXPECTED_COUNT or metrics["split"] != "test"
                or metrics["evaluation_version"] != "evaluation_v2"
                or {t: row["count"] for t, row in metrics["per_task"].items()} != completion["task_counts"]):
            raise ValueError("Final-test scoring is incomplete")
        verify_freeze(ROOT)
        generation = read_json(run_dir / "runner_summary.json")
        result.update(status="TASK_A_FINAL_TEST_PASSED", scope="All 1007 test examples; frozen Task A baseline",
                      final_test_frozen=True, gpu_execution=generation["gpu_execution"],
                      task_counts=completion["task_counts"], completed_count=EXPECTED_COUNT,
                      freeze_manifest_sha256=FREEZE_SHA256, scoring_directory=scoring_dir.name,
                      blank_raw_answers=generation["blank_raw_answers"],
                      empty_extracted_answers=generation["empty_extracted_answers"],
                      truncated_outputs=generation["truncated_outputs"], max_input_tokens=generation["max_input_tokens"],
                      generation_wall_seconds=generation["generation_wall_seconds"], gpu_memory=generation["gpu_memory"],
                      per_task=metrics["per_task"], macro_primary_score=metrics["macro_primary_score"],
                      interpretation="Complete final-test execution; all incorrect/empty/truncated outputs remain in the denominator")
        save_json(scoring_dir / "scoring_summary.json", result)
        save_json(run_dir / "final_summary.json", result)
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    except BaseException as exc:
        result.update(status="scoring_failed", error_type=type(exc).__name__, error=str(exc))
        save_json(scoring_dir / "scoring_summary.json", result)
        raise
    finally:
        archive = shutil.make_archive(str(run_dir) + "_scored_" + timestamp, "zip", root_dir=run_dir)
        namespace["TASK_A_TEST_RESULT_ZIP"] = archive
        print("Result ZIP:", archive, flush=True)
        if not args.no_download:
            try:
                from google.colab import files
                files.download(archive)
            except Exception as exc:
                print("ZIP saved. Retry files.download(TASK_A_TEST_RESULT_ZIP):", str(exc), flush=True)


if __name__ == "__main__":
    main(globals())
