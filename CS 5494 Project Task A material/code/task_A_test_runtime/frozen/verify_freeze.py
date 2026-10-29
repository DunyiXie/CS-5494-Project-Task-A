"""Verify the portable Task A release; default checks need only Python stdlib."""

import argparse
import hashlib
import importlib
from importlib.metadata import version
import json
from pathlib import Path, PurePosixPath
import sys


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(root, check_runtime=False):
    root = Path(root).resolve()
    manifest = json.loads((root / "freeze_manifest.json").read_text(encoding="utf-8"))
    checks = []

    def require(condition, label):
        if not condition:
            raise ValueError(label)
        checks.append(label)

    require(manifest["scope"] == "Task A only", "Task_A_scope")
    expected = set(manifest["files"])
    actual = {p.relative_to(root).as_posix() for p in root.rglob("*")
              if p.is_file() and "__pycache__" not in p.parts and p.name != "freeze_manifest.json"}
    require(actual == expected, "exact_release_file_inventory")
    for name, entry in manifest["files"].items():
        relative = PurePosixPath(name)
        require(not relative.is_absolute() and ".." not in relative.parts
                and "\\" not in name and ":" not in name, "safe_path:" + name)
        path = root.joinpath(*relative.parts)
        require(path.resolve().is_relative_to(root) and not path.is_symlink(), "within_release:" + name)
        require(path.stat().st_size == entry["bytes"] and digest(path) == entry["sha256"], "sha256:" + name)
    require(not any(PurePosixPath(p).name.casefold().startswith("test_answer")
                    or PurePosixPath(p).name.casefold() == "test.csv" for p in expected), "no_test_files_in_release")
    config = json.loads((root / "config_frozen.json").read_text(encoding="utf-8"))
    require((root / "config_frozen.json").read_bytes() ==
            (root / "project/common/config.json").read_bytes(), "explicit_and_default_configs_identical")
    require(config["status"] == "task_A_final_frozen" and config["final_test_frozen"] is True,
            "Task_A_final_test_flag")
    require(config["freeze"]["freeze_id"] == manifest["freeze_id"], "freeze_id_matches")
    source = json.loads((root / "evidence/source_config_before_freeze.json").read_text(encoding="utf-8"))
    behavior = {k: v for k, v in config.items() if k != "freeze"}
    behavior["status"], behavior["final_test_frozen"] = source["status"], source["final_test_frozen"]
    require(behavior == source, "all_inference_behavior_unchanged_from_validation")
    contract = json.loads((root / "method_contract.json").read_text(encoding="utf-8"))
    require(contract["method"] == "baseline" and contract["n_shot"] == 0
            and contract["adapter"] is None and contract["weights_updated"] is False,
            "unchanged_base_model_without_demos_or_adapter")
    require(contract["expected_test_rows"] == 1007 and
            contract["references_access"] == "Only after all test predictions have been generated and saved",
            "final_test_protocol_fixed")
    require(manifest["test_inputs_read_during_freeze"] is False and
            manifest["test_references_read_during_freeze"] is False and
            manifest["final_test_inference"] == "not_run", "freeze_is_not_a_test_run")
    for name in list(sys.modules):
        if name == "project" or name.startswith("project."):
            del sys.modules[name]
    sys.path.insert(0, str(root))
    inference = importlib.import_module("project.common.inference")
    evaluation = importlib.import_module("project.common.evaluation")
    require(Path(inference.__file__).resolve() == root / "project/common/inference.py" and
            Path(evaluation.__file__).resolve() == root / "project/common/evaluation.py",
            "imports_resolve_to_frozen_sources")
    inference.validate_config(config)
    require(evaluation.VERSION == config["freeze"]["evaluation_version"] == "evaluation_v2",
            "evaluation_v2_fixed")
    require(str(evaluation.ABS_TOL) == "1/1000000" and evaluation.REL_TOL == 0,
            "numeric_tolerance_fixed")
    for path in root.rglob("*.py"):
        compile(path.read_bytes(), str(path), "exec")
    require(True, "all_frozen_Python_sources_compile")
    require(inference.input_only({"Sample_id": "synthetic_check", "Task": "qa", "Question": "Q?",
                                  "Context": "", "Answer": "PRIVATE"}).keys() ==
            {"Sample_id", "Task", "Question", "Context"}, "target_answer_fields_excluded")
    runtime = {"checked": False, "reason": "Default verification does not load torch or a model"}
    if check_runtime:
        lock = json.loads((root / "environment_lock.json").read_text(encoding="utf-8"))
        for name, wanted in lock["critical_packages"].items():
            require(version(name) == wanted, "runtime_package:" + name)
        import torch
        require(str(torch.__version__) == lock["torch"], "runtime_torch")
        require(torch.version.cuda == lock["cuda"] and torch.cuda.is_available(), "runtime_CUDA")
        require(torch.cuda.get_device_name(0) == lock["gpu"], "runtime_GPU")
        runtime = {"checked": True, "model_loaded": False, "gpu": torch.cuda.get_device_name(0)}
    return {"status": "TASK_A_FROZEN_RELEASE_VERIFIED", "freeze_id": manifest["freeze_id"],
            "manifest_sha256": digest(root / "freeze_manifest.json"), "files_verified": len(expected),
            "checks_passed": len(checks), "checks": checks, "runtime": runtime,
            "test_data_read": False, "model_generation_performed": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-runtime", action="store_true", help="Also verify the actual Colab environment, without loading weights")
    args = parser.parse_args()
    print(json.dumps(verify(Path(__file__).resolve().parent, args.check_runtime), ensure_ascii=False, indent=2))
