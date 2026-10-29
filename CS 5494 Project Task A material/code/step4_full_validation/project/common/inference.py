"""Shared prompt preparation, reference-blind inference and optional GPU backend.

Importing this module does not import torch, transformers or PEFT. CPU checks can
exercise the orchestration with a clearly labelled simulated backend.
"""

from collections import Counter
from dataclasses import dataclass
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import random
import re
import time

from .data_utils import TASKS, index_by_id, load_inputs, read_csv, write_predictions
from .evaluation import VERSION as EVALUATION_VERSION, extract_answer

VERSION = "inference_v1"
COMMON = Path(__file__).resolve().parent
INPUT_FIELDS = ("Sample_id", "Task", "Question", "Context")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def validate_config(config):
    if config.get("model_id") != "Qwen/Qwen3-1.7B" or config.get("tokenizer_id") != config["model_id"]:
        raise ValueError("This shared configuration expects Qwen/Qwen3-1.7B and its tokenizer")
    revision = config.get("model_revision", "")
    if not re.fullmatch(r"[0-9a-f]{40}", revision) or config.get("tokenizer_revision") != revision:
        raise ValueError("Model and tokenizer must use the same pinned 40-character commit")
    if config.get("dtype") != "float16" or config.get("quantization") is not None or config.get("batch_size") != 1:
        raise ValueError("Comparison inference requires unquantized FP16 and batch_size=1")
    generation = config["generation"]
    if generation.get("do_sample") is not False or generation.get("num_beams") not in (None, 1):
        raise ValueError("Shared decoding must be greedy")
    maximum = generation.get("max_new_tokens")
    lengths = config["input_processing"]
    if type(maximum) is not int or maximum <= 0 or type(lengths["max_input_tokens"]) is not int or lengths["max_input_tokens"] <= 0:
        raise ValueError("Token budgets must be positive integers")
    if lengths["max_input_tokens"] + maximum > lengths["total_token_budget"]:
        raise ValueError("Input and output budgets exceed the total token budget")
    if lengths["padding_side"] != "left" or lengths.get("add_special_tokens_after_chat_template") is not False:
        raise ValueError("Use left padding and do not add special tokens twice")
    if lengths.get("truncation") != "none; reject over-budget input":
        raise ValueError("Target truncation is not supported by this policy")
    if config["chat_template"] != {"enable_thinking": False, "add_generation_prompt": True}:
        raise ValueError("The shared prompt must disable thinking and add a generation prompt")
    if config.get("attention_implementation") != "sdpa":
        raise ValueError("The current comparison uses SDPA attention")
    eos = generation.get("eos_token_id")
    eos = [eos] if type(eos) is int else eos
    if not isinstance(eos, list) or not eos or any(type(item) is not int for item in eos):
        raise ValueError("Explicit EOS token IDs are required")
    for task in TASKS:
        if not config["system_prompts"].get(task) or not config["user_templates"].get(task):
            raise ValueError(f"Missing task prompt: {task}")
    return config


def input_only(row):
    """Explicit allowlist: Answer/Accepted_answers never reach target prompts."""
    clean = {key: row.get(key, "") for key in INPUT_FIELDS}
    if any(not isinstance(value, str) for value in clean.values()):
        raise ValueError("Input fields must be strings")
    index_by_id([clean])
    if clean["Task"] not in TASKS or not clean["Question"].strip():
        raise ValueError(f"Invalid task or question: {clean['Sample_id']}")
    if clean["Task"] == "rag" and not clean["Context"].strip():
        raise ValueError(f"Missing RAG context: {clean['Sample_id']}")
    return clean


def read_inference_inputs(path, config, split="validation"):
    validate_config(config)
    if split not in ("validation", "test"):
        raise ValueError("Choose validation or test mode explicitly")
    if split == "test" and config.get("final_test_frozen") is not True:
        raise ValueError("Freeze the final test configuration before opening test inputs")
    return [input_only(row) for row in load_inputs(path, allow_test=split == "test")]


@dataclass
class TrainingPool:
    rows: dict
    file_sha256: str


def load_training_pool(path):
    policy = json.loads((COMMON / "inference_policy.json").read_text(encoding="utf-8"))
    if Path(path).name.casefold() == "test.csv" or Path(path).name.casefold().startswith("test_answer"):
        raise ValueError("Test data cannot be used as training demonstrations")
    # Reject a wrong subset before reading any labels from it.
    if sha256(path) != policy["training_split_sha256"]:
        raise ValueError("Demonstrations must come from the frozen train_split.csv")
    rows = read_csv(path, [*INPUT_FIELDS, "Answer"])
    if len(rows) != policy["training_split_rows"]:
        raise ValueError("Unexpected training-subset size")
    for row in rows:
        input_only(row)
        if not row["Answer"].strip():
            raise ValueError("Empty training demonstration answer")
    return TrainingPool(index_by_id(rows), sha256(path))


def user_message(row, config):
    clean = input_only(row)
    return config["user_templates"][clean["Task"]].format(question=clean["Question"], context=clean["Context"])


def prepare_prompt(row, tokenizer, config, demo_ids=(), training_pool=None):
    target = input_only(row)
    requested = list(demo_ids)
    if len(requested) != len(set(requested)):
        raise ValueError("Duplicate demonstration IDs")
    if requested and not isinstance(training_pool, TrainingPool):
        raise ValueError("A verified training pool is required for demonstrations")
    demonstrations = []
    for sample_id in requested:
        if sample_id == target["Sample_id"] or sample_id not in training_pool.rows:
            raise ValueError(f"Target or non-training demonstration ID: {sample_id}")
        demo = training_pool.rows[sample_id]
        if demo["Task"] != target["Task"]:
            raise ValueError("Demonstration task must match the target task")
        demonstrations.append(demo)
    used = list(requested)
    while True:
        messages = [{"role": "system", "content": config["system_prompts"][target["Task"]]}]
        for demo in demonstrations:
            messages.extend([{"role": "user", "content": user_message(demo, config)},
                             {"role": "assistant", "content": demo["Answer"]}])
        messages.append({"role": "user", "content": user_message(target, config)})
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, **config["chat_template"])
        encoded = tokenizer(prompt, add_special_tokens=False, truncation=False)
        count = len(encoded["input_ids"])
        if count <= config["input_processing"]["max_input_tokens"]:
            return {"Sample_id": target["Sample_id"], "Task": target["Task"], "messages": messages,
                    "prompt": prompt, "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                    "input_tokens": count, "requested_demo_ids": requested, "used_demo_ids": used,
                    "dropped_demo_ids": requested[len(used):], "actual_n": len(used)}
        if not demonstrations:
            raise ValueError(f"Target input exceeds token budget: {target['Sample_id']} ({count} tokens); no target text was truncated")
        demonstrations.pop()
        used.pop()


def generation_record(token_ids, raw_answer, raw_tokens_text, input_tokens, seconds, config):
    limit = config["generation"]["max_new_tokens"]
    if len(token_ids) > limit or seconds < 0:
        raise ValueError("Invalid generation length or timing")
    eos = config["generation"]["eos_token_id"]
    eos = [eos] if type(eos) is int else eos
    ended = bool(token_ids) and token_ids[-1] in eos
    return {"raw_answer": raw_answer, "raw_new_tokens_text": raw_tokens_text,
            "output_token_ids": token_ids, "input_tokens": input_tokens,
            "output_tokens_including_eos": len(token_ids), "generation_seconds": seconds,
            "hit_output_limit": len(token_ids) == limit, "ended_with_eos": ended,
            "truncated": not ended, "stop_reason": "eos" if ended else "length" if len(token_ids) == limit else "no_eos"}


def run_inference(inputs, backend, config, output_dir, method="baseline", demo_ids_by_sample=None, training_pool=None):
    """Generate using target input fields only. No reference file is accepted."""
    validate_config(config)
    rows = [input_only(row) for row in inputs]
    ids = index_by_id(rows, "inputs")
    if not rows:
        raise ValueError("Cannot infer an empty input set")
    if method not in ("baseline", "nshot", "sft"):
        raise ValueError("Unknown method")
    demos = demo_ids_by_sample or {}
    if set(demos) - set(ids):
        raise ValueError("Demonstration map contains unknown target IDs")
    if method != "nshot" and any(demos.values()):
        raise ValueError("Only the nshot method may add demonstrations")
    if bool(backend.metadata.get("adapter")) != (method == "sft"):
        raise ValueError("SFT requires an adapter; baseline/nshot require an unchanged base model")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    import csv
    with (output_dir / "inputs.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=INPUT_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    save_json(output_dir / "config_snapshot.json", config)
    save_json(output_dir / "environment.json", backend.metadata)
    code_hashes = {name: sha256(COMMON / name) for name in
                   ["inference.py", "inference_policy.json", "evaluation.py", "evaluation_config.json", "data_utils.py"]}
    summary = {"status": "running", "stage": "preflight", "inference_version": VERSION,
               "evaluation_version": EVALUATION_VERSION, "method": method,
               "requested_count": len(rows), "completed_count": 0,
               "task_counts": dict(Counter(row["Task"] for row in rows)),
               "input_sha256": sha256(output_dir / "inputs.csv"), "code_sha256": code_hashes,
               "config_sha256": sha256(output_dir / "config_snapshot.json"),
               "training_pool_sha256": training_pool.file_sha256 if training_pool else None,
               "backend_kind": backend.metadata["backend_kind"],
               "timing_scope": "Synchronized per-example generation excludes tokenization, loading, preflight and warm-up",
               "memory_scope": "GPU session memory includes any other objects retained in the notebook"}
    save_json(output_dir / "run_summary.json", summary)
    records, prepared = [], []
    started = time.perf_counter()
    try:
        with (output_dir / "prompts.jsonl").open("w", encoding="utf-8") as prompts:
            for row in rows:
                item = prepare_prompt(row, backend.tokenizer, config, demos.get(row["Sample_id"], ()), training_pool)
                prepared.append(item)
                prompts.write(json.dumps(item, ensure_ascii=False) + "\n")
                prompts.flush()
        summary.update(stage="warmup", max_input_tokens=max(item["input_tokens"] for item in prepared),
                       preflight_seconds=time.perf_counter() - started,
                       actual_n_counts=dict(Counter(item["actual_n"] for item in prepared)))
        save_json(output_dir / "run_summary.json", summary)
        warmup_started = time.perf_counter()
        backend.warmup(prepared[0])
        summary["warmup_seconds"] = time.perf_counter() - warmup_started
        backend.start_measurement()
        summary["stage"] = "generation"
        save_json(output_dir / "run_summary.json", summary)
        inference_started = time.perf_counter()
        with (output_dir / "raw_outputs.jsonl").open("w", encoding="utf-8") as trace:
            for index, item in enumerate(prepared, start=1):
                generated = backend.generate(item)
                if generated["input_tokens"] != item["input_tokens"] or type(generated["truncated"]) is not bool:
                    raise ValueError("Backend token count or truncation flag is inconsistent")
                extracted = extract_answer(item["Task"], generated["raw_answer"], generated["truncated"])
                record = {"Sample_id": item["Sample_id"], "Task": item["Task"], **generated,
                          "Answer": extracted["answer"], "extraction_status": extracted["status"],
                          "prompt_sha256": item["prompt_sha256"], "requested_demo_ids": item["requested_demo_ids"],
                          "used_demo_ids": item["used_demo_ids"], "dropped_demo_ids": item["dropped_demo_ids"],
                          "actual_n": item["actual_n"]}
                trace.write(json.dumps(record, ensure_ascii=False) + "\n")
                trace.flush()
                records.append(record)
                summary["completed_count"] = len(records)
                save_json(output_dir / "run_summary.json", summary)
                print(f"[{index}/{len(rows)}] {item['Task']} {item['Sample_id']} | {generated['generation_seconds']:.3f}s | answer={extracted['answer']!r}", flush=True)
        summary.update(generation_wall_seconds=time.perf_counter() - inference_started,
                       summed_generation_seconds=sum(item["generation_seconds"] for item in records),
                       blank_raw_answers=sum(not item["raw_answer"].strip() for item in records),
                       empty_extracted_answers=sum(not item["Answer"] for item in records),
                       truncated_outputs=sum(item["truncated"] for item in records),
                       output_limit_hits=sum(item["hit_output_limit"] for item in records),
                       extraction_status_counts=dict(Counter(item["extraction_status"] for item in records)),
                       gpu_memory=backend.memory_stats(), total_pipeline_seconds=time.perf_counter() - started)
        if set(index_by_id(records)) != set(ids):
            raise ValueError("Generated IDs differ from input IDs")
        temporary_predictions = output_dir / "predictions.complete.tmp"
        write_predictions(temporary_predictions, records)
        temporary_predictions.replace(output_dir / "predictions.csv")
        summary.update(status="complete", stage="complete", predictions_sha256=sha256(output_dir / "predictions.csv"))
        save_json(output_dir / "run_summary.json", summary)
        return summary
    except BaseException as exc:
        summary.update(status="failed", completed_count=len(records), error_type=type(exc).__name__, error=str(exc),
                       total_pipeline_seconds=time.perf_counter() - started)
        save_json(output_dir / "run_summary.json", summary)
        raise


def origin_matches(name, commit, config):
    path = str(name or "").replace("\\", "/").rstrip("/")
    return ((path == config["model_id"] and commit == config["model_revision"]) or
            path.endswith("/models--Qwen--Qwen3-1.7B/snapshots/" + config["model_revision"]))


def reusable_model_reason(model, tokenizer, config):
    """Check recorded origin and actual precision/device; no weight mutation."""
    if model is None or tokenizer is None:
        return "model or tokenizer missing"
    if getattr(model, "peft_config", None) or getattr(model, "is_quantized", False):
        return "adapter or quantization is attached"
    metadata = getattr(model, "config", None)
    if getattr(metadata, "model_type", None) != "qwen3" or not origin_matches(
            getattr(metadata, "_name_or_path", None), getattr(metadata, "_commit_hash", None), config):
        return "model origin is not the pinned snapshot"
    if not origin_matches(getattr(tokenizer, "name_or_path", None), getattr(tokenizer, "init_kwargs", {}).get("_commit_hash"), config):
        return "tokenizer origin is not the pinned snapshot"
    if getattr(metadata, "_attn_implementation", None) != config["attention_implementation"]:
        return "attention implementation differs"
    for parameter in model.parameters():
        if str(parameter.device) != "cuda:0" or (parameter.is_floating_point() and str(parameter.dtype) != "torch.float16"):
            return "parameters are not unquantized FP16 on cuda:0"
    if any("Linear4bit" in type(layer).__name__ or "Linear8bit" in type(layer).__name__ for layer in model.modules()):
        return "quantized layers are present"
    return None


def adapter_metadata(path, config):
    path = Path(path).resolve()
    adapter_config = json.loads((path / "adapter_config.json").read_text(encoding="utf-8"))
    provenance = json.loads((path / "adapter_provenance.json").read_text(encoding="utf-8"))
    if provenance.get("base_model_id") != config["model_id"] or provenance.get("base_model_revision") != config["model_revision"]:
        raise ValueError("Adapter provenance must identify the pinned base model and commit")
    if not origin_matches(adapter_config.get("base_model_name_or_path"), adapter_config.get("revision") or config["model_revision"], config):
        raise ValueError("Adapter configuration names a different base model")
    weights = sorted(path.glob("adapter_model*.safetensors"))
    if not weights:
        raise ValueError("Adapter safetensors weights are required")
    return {"path": str(path), "provenance": provenance,
            "sha256": {file.name: sha256(file) for file in [path / "adapter_config.json", path / "adapter_provenance.json", *weights]}}


class TorchBackend:
    """Actual CUDA FP16 generation. Optional PEFT is imported only for SFT."""

    def __init__(self, config, reuse_model=None, reuse_tokenizer=None, adapter_path=None):
        validate_config(config)
        import torch
        from huggingface_hub import snapshot_download
        from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
        if not torch.cuda.is_available():
            raise RuntimeError("Enable a GPU runtime in Colab before running step 4.2")
        self.torch, self.config = torch, config
        adapter = adapter_metadata(adapter_path, config) if adapter_path else None
        started = time.perf_counter()
        random.seed(config["seed"])
        torch.manual_seed(config["seed"])
        torch.cuda.manual_seed_all(config["seed"])
        snapshot = Path(snapshot_download(config["model_id"], revision=config["model_revision"],
                        allow_patterns=["*.json", "*.safetensors", "*.jinja", "merges.txt", "vocab.json", "LICENSE*"]))
        reason = reusable_model_reason(reuse_model, reuse_tokenizer, config)
        reused = reason is None and adapter is None
        if reused:
            self.model, self.tokenizer = reuse_model, reuse_tokenizer
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True)
            self.model = AutoModelForCausalLM.from_pretrained(str(snapshot), local_files_only=True,
                             dtype=torch.float16, device_map={"": 0}, attn_implementation=config["attention_implementation"])
        if adapter:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, adapter["path"], is_trainable=False, autocast_adapter_dtype=False)
            self.model.to(dtype=torch.float16)
        elif reusable_model_reason(self.model, self.tokenizer, config) is not None:
            raise RuntimeError("Loaded model does not match the pinned FP16 configuration")
        self.model.eval()
        self.model.requires_grad_(False)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id != config["generation"]["pad_token_id"]:
            raise ValueError("Tokenizer padding token differs from the shared configuration")
        eos_ids = config["generation"]["eos_token_id"]
        eos_ids = [eos_ids] if type(eos_ids) is int else eos_ids
        if self.tokenizer.eos_token_id not in eos_ids:
            raise ValueError("Tokenizer EOS token differs from the shared configuration")
        generation_values = {key: value for key, value in config["generation"].items() if value is not None and not key.startswith("_")}
        self.generation = GenerationConfig(**generation_values)
        self.generation.num_beams = 1
        self.generation.return_dict_in_generate = False
        self.generation.output_scores = False
        names = ["transformers", "accelerate", "huggingface-hub", "tokenizers", "safetensors", "bitsandbytes", "peft"]
        packages = {}
        for name in names:
            try:
                packages[name] = version(name)
            except PackageNotFoundError:
                packages[name] = None
        self.metadata = {"backend_kind": "torch_cuda", "model_id": config["model_id"],
                         "model_revision": config["model_revision"], "tokenizer_revision": config["tokenizer_revision"],
                         "snapshot_path": str(snapshot), "model_reused": reused, "reload_reason": None if reused else reason,
                         "model_setup_seconds": time.perf_counter() - started,
                         "snapshot_config_sha256": {name: sha256(snapshot / name) for name in
                             ["config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja"] if (snapshot / name).is_file()},
                         "dtype": "float16", "quantized": False, "adapter": adapter,
                         "attention_implementation": getattr(self.model.config, "_attn_implementation", None),
                         "parameter_devices": sorted({str(p.device) for p in self.model.parameters()}),
                         "floating_parameter_dtypes": sorted({str(p.dtype) for p in self.model.parameters() if p.is_floating_point()}),
                         "model_parameter_and_buffer_gib": self.model.get_memory_footprint() / 1024**3,
                         "torch": torch.__version__, "cuda": torch.version.cuda, "packages": packages,
                         "gpu": torch.cuda.get_device_name(0),
                         "gpu_capacity_gib": torch.cuda.get_device_properties(0).total_memory / 1024**3,
                         "generation_effective": self.generation.to_dict(),
                         "reuse_verification": "Origin metadata, precision, device and absence of adapters/quantization; model weights are not hashed in GPU memory"}

    def encoded(self, prepared):
        batch = self.tokenizer(prepared["prompt"], return_tensors="pt", add_special_tokens=False, truncation=False)
        if batch["input_ids"].shape[1] != prepared["input_tokens"]:
            raise ValueError("Prompt tokenization changed after preflight")
        return batch.to(self.model.get_input_embeddings().weight.device)

    def warmup(self, prepared):
        batch = self.encoded(prepared)
        with self.torch.inference_mode():
            result = self.model.generate(**batch, generation_config=self.generation, max_new_tokens=8)
        del result, batch
        self.torch.cuda.synchronize()

    def start_measurement(self):
        self.torch.cuda.synchronize()
        self.torch.cuda.reset_peak_memory_stats()

    def generate(self, prepared):
        batch = self.encoded(prepared)
        self.torch.cuda.synchronize()
        started = time.perf_counter()
        with self.torch.inference_mode():
            generated = self.model.generate(**batch, generation_config=self.generation)
        self.torch.cuda.synchronize()
        seconds = time.perf_counter() - started
        ids = generated[0, prepared["input_tokens"]:].tolist()
        raw = self.tokenizer.decode(ids, skip_special_tokens=True).strip()
        raw_tokens = self.tokenizer.decode(ids, skip_special_tokens=False)
        return generation_record(ids, raw, raw_tokens, prepared["input_tokens"], seconds, self.config)

    def memory_stats(self):
        cuda = self.torch.cuda
        return {"peak_allocated_gib": cuda.max_memory_allocated() / 1024**3,
                "peak_reserved_gib": cuda.max_memory_reserved() / 1024**3,
                "ending_allocated_gib": cuda.memory_allocated() / 1024**3}
