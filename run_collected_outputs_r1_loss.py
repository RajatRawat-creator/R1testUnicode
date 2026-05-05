#!/usr/bin/env python3
import gc
import json
import os
from pathlib import Path

os.environ.setdefault("DISABLE_KERNEL_MAPPING", "1")
os.environ.setdefault("HF_HUB_DISABLE_KERNELS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "deepseek-ai/DeepSeek-R1"
DATASETS_DIR = "collected_outputs"
OUT_DIR = "outputs_r1_collected"

LIMIT_PER_FILE = None   # set to 2 for smoke test, None for all
MAX_LENGTH = 32768
DTYPE = "bfloat16"
DEVICE_MAP = "auto"
TRUST_REMOTE_CODE = True
LOCAL_FILES_ONLY = False
MAX_MEMORY_PER_GPU = "130GiB"
MAX_MEMORY_CPU = "200GiB"


def dtype_from_string(dtype):
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[dtype]


def build_max_memory():
    if not torch.cuda.is_available():
        return {"cpu": MAX_MEMORY_CPU}
    mem = {i: MAX_MEMORY_PER_GPU for i in range(torch.cuda.device_count())}
    mem["cpu"] = MAX_MEMORY_CPU
    return mem


def get_input_device(model):
    if hasattr(model, "hf_device_map") and isinstance(model.hf_device_map, dict):
        for dev in model.hf_device_map.values():
            if isinstance(dev, int):
                return torch.device(f"cuda:{dev}")
            if isinstance(dev, str) and dev.startswith("cuda"):
                return torch.device(dev)
    return next(model.parameters()).device


def load_jsonl(path, limit=None):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if limit is not None and len(rows) >= limit:
                break
            try:
                item = json.loads(line)
            except Exception:
                continue

            q = item.get("question") or item.get("prompt") or item.get("input") or ""
            a = (
                item.get("response")
                or item.get("answer")
                or item.get("output")
                or item.get("completion")
                or ""
            )

            if q and a:
                rows.append({
                    "question": q,
                    "answer": a,
                    "document": f"Problem: {q}\nSolution: {a}",
                    "raw": item,
                })
    return rows


class LossWrapper:
    def __init__(self, model, tokenizer, max_length):
        self.model = model
        self.tokenizer = tokenizer
        self.device = get_input_device(model)
        self.max_length = max_length

    def loss(self, document):
        marker = "Solution:"
        idx = document.find(marker)
        if idx == -1:
            return None

        prompt_text = document[: idx + len(marker)]
        answer_text = document[idx + len(marker):].lstrip()
        if not answer_text.strip():
            return None

        prompt_ids = self.tokenizer(prompt_text, add_special_tokens=True)["input_ids"]
        answer_ids = self.tokenizer(answer_text, add_special_tokens=False)["input_ids"]

        if not answer_ids:
            return None

        orig_prompt_len = len(prompt_ids)
        orig_answer_len = len(answer_ids)
        orig_total = orig_prompt_len + orig_answer_len

        if orig_total > self.max_length:
            if len(answer_ids) >= self.max_length:
                answer_ids = answer_ids[-(self.max_length - 1):]
                prompt_ids = prompt_ids[:1]
            else:
                keep_prompt = self.max_length - len(answer_ids)
                prompt_ids = prompt_ids[-keep_prompt:]

        input_ids_list = prompt_ids + answer_ids
        labels_list = [-100] * len(prompt_ids) + answer_ids

        input_ids = torch.tensor([input_ids_list], device=self.device)
        labels = torch.tensor([labels_list], device=self.device)
        attention_mask = torch.ones_like(input_ids)

        with torch.inference_mode():
            out = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                use_cache=False,
            )

        if out.loss is None or not torch.isfinite(out.loss):
            return None

        return {
            "loss": float(out.loss.detach().cpu()),
            "orig_prompt_tokens": orig_prompt_len,
            "orig_answer_tokens": orig_answer_len,
            "orig_total_tokens": orig_total,
            "used_prompt_tokens": len(prompt_ids),
            "used_answer_tokens": len(answer_ids),
            "used_total_tokens": len(input_ids_list),
            "trimmed": bool(orig_total > self.max_length),
        }


def safe_name(path):
    return str(path).replace("/", "__").replace(" ", "_").replace("(", "").replace(")", "")


def main():
    datasets_dir = Path(DATASETS_DIR)
    out_dir = Path(OUT_DIR)
    out_dir.mkdir(exist_ok=True, parents=True)

    files = sorted(datasets_dir.rglob("*.jsonl"))
    if not files:
        raise RuntimeError(f"No .jsonl files found under {datasets_dir}")

    print(f"[info] Found {len(files)} jsonl files")
    for f in files:
        print(f"  - {f}")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=TRUST_REMOTE_CODE,
        local_files_only=LOCAL_FILES_ONLY,
    )
    tokenizer.truncation_side = "left"
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map=DEVICE_MAP,
        dtype=dtype_from_string(DTYPE),
        trust_remote_code=False,
        local_files_only=LOCAL_FILES_ONLY,
        low_cpu_mem_usage=True,
        max_memory=build_max_memory(),
    )
    model.config.use_cache = False
    model.eval()

    wrapper = LossWrapper(model, tokenizer, MAX_LENGTH)

    all_results = {
        "model_name": MODEL_NAME,
        "datasets_dir": str(datasets_dir),
        "max_length": MAX_LENGTH,
        "limit_per_file": LIMIT_PER_FILE,
        "results": {},
    }

    for path in files:
        rel = str(path.relative_to(datasets_dir))
        print(f"\n[file] {rel}")

        samples = load_jsonl(path, LIMIT_PER_FILE)
        rows = []
        losses = []

        for i, sample in enumerate(samples):
            meta = wrapper.loss(sample["document"])
            row = {
                "idx": i,
                "question": sample["question"],
                "answer": sample["answer"],
                "ok": meta is not None,
                "loss": None,
            }
            if meta is not None:
                row.update(meta)
                losses.append(meta["loss"])

            rows.append(row)

            if (i + 1) % 10 == 0:
                print(f"  done {i+1}/{len(samples)}")

        summary = {
            "count": len(rows),
            "valid_count": len(losses),
            "mean_loss": float(np.mean(losses)) if losses else None,
            "std_loss": float(np.std(losses)) if losses else None,
        }

        all_results["results"][rel] = {
            "summary": summary,
            "rows": rows,
        }

        per_file_out = out_dir / f"{safe_name(rel)}__r1_losses.json"
        with open(per_file_out, "w", encoding="utf-8") as f:
            json.dump(all_results["results"][rel], f, indent=2, ensure_ascii=False)

        print(f"  valid {summary['valid_count']}/{summary['count']}")
        print(f"  mean loss: {summary['mean_loss']}")

    combined_out = out_dir / "DeepSeek-R1__collected_outputs_losses.json"
    with open(combined_out, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"\nSaved combined output to: {combined_out}")

    del model, tokenizer, wrapper
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
