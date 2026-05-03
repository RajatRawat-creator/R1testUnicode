#!/usr/bin/env python3
# Single-model O1 Unicode-vs-ASCII loss cache script
# - Same style as the previous single_model_loss_cache.py
# - No argparse: edit settings below, then run `python run_o1_loss_r1.py`
# - Computes answer-only average NLL for DeepSeek-R1 on:
#     1. o1 Unicode
#     2. o1 ASCII/default, with non-ASCII chars re-escaped to literal \uXXXX

import gc
import json
import os
import re
from pathlib import Path

import numpy as np
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

# ============================================================
# Compatibility patch for DeepSeek remote modeling code
# Needed for newer Transformers versions where is_torch_fx_available
# is no longer exposed, but DeepSeek's remote modeling_deepseek.py imports it.
# ============================================================

import transformers.utils.import_utils as _tf_import_utils

if not hasattr(_tf_import_utils, "is_torch_fx_available"):
    def _is_torch_fx_available():
        return False
    _tf_import_utils.is_torch_fx_available = _is_torch_fx_available


# =========================
# HARD-CODED SETTINGS
# =========================

MODEL_NAME = "deepseek-ai/DeepSeek-R1"
DATASETS_DIR = "O1_UnicodeandASCII"
OUT_DIR = "./outputs_r1"

# For smoke test, use 2.
# For real run, change to 200.
LIMIT_PER_DATASET = 200

MAX_LENGTH = 32768

# Choose one of: "float16", "bfloat16", "float32"
DTYPE = "bfloat16"

# Usually keep as "auto"
DEVICE_MAP = "auto"

TRUST_REMOTE_CODE = True
LOCAL_FILES_ONLY = False
DEBUG = False

# This is how your earlier working script loaded DeepSeek-R1.
# Keep True if that earlier run worked with bitsandbytes.
LOAD_IN_8BIT = False

# Optional memory cap for multi-GPU auto placement.
# On 8x H200 this is reasonable. If using 1 GPU smoke test,
# it will just cap that one visible GPU.
MAX_MEMORY_PER_GPU = "130GiB"
MAX_MEMORY_CPU = "200GiB"

# =========================


FILES_MAP = {
    "o1 (OMI, Unicode)": {
        "file": "o1_openmath__responses_unicode.jsonl",
        "transform": None,
    },
    "o1 (OMI, ASCII)": {
        "file": "o1__responses_default.jsonl",
        "transform": "escape",
    },
}


def safe_name(s: str) -> str:
    return "".join(c if (c.isalnum() or c in "-_+.") else "_" for c in s)[:180]


def dtype_from_string(dtype: str):
    if dtype == "float16":
        return torch.float16
    if dtype == "bfloat16":
        return torch.bfloat16
    if dtype == "float32":
        return torch.float32
    raise ValueError(f"Unknown DTYPE={dtype}")


def build_max_memory():
    if MAX_MEMORY_PER_GPU is None:
        return None
    if not torch.cuda.is_available():
        return {"cpu": MAX_MEMORY_CPU}
    max_memory = {i: MAX_MEMORY_PER_GPU for i in range(torch.cuda.device_count())}
    max_memory["cpu"] = MAX_MEMORY_CPU
    return max_memory


def get_input_device(model):
    """Prefer a CUDA device that hosts LM embeddings."""
    if hasattr(model, "hf_device_map") and isinstance(model.hf_device_map, dict):
        dmap = model.hf_device_map

        preferred_keys = [
            "model.embed_tokens",
            "model.model.embed_tokens",
            "transformer.wte",
            "model.tok_embeddings",
            "model.model.tok_embeddings",
        ]
        for k in preferred_keys:
            if k in dmap:
                dev = dmap[k]
                if dev == "disk":
                    return torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
                if isinstance(dev, int):
                    return torch.device(f"cuda:{dev}")
                return torch.device(dev)

        for dev in dmap.values():
            if isinstance(dev, int):
                return torch.device(f"cuda:{dev}")
            if isinstance(dev, str) and dev.startswith("cuda"):
                return torch.device(dev)

        return torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    return next(model.parameters()).device


def get_model_ctx_limit(model, tokenizer, default=32768):
    vals = []
    for attr in ["n_positions", "max_position_embeddings"]:
        v = getattr(model.config, attr, None)
        if isinstance(v, int) and v > 0:
            vals.append(v)

    tv = getattr(tokenizer, "model_max_length", None)
    if isinstance(tv, int) and 0 < tv < 100000:
        vals.append(tv)

    return min(vals) if vals else default


def resolve_dataset_file(datasets_dir: Path, filename: str) -> Path:
    p = datasets_dir / filename
    if p.exists():
        return p

    stem = filename[:-6] if filename.endswith(".jsonl") else filename
    pattern = re.compile(rf"^{re.escape(stem)}(\s*\(\d+\))?\.jsonl$")
    matches = [x for x in datasets_dir.iterdir() if x.is_file() and pattern.match(x.name)]
    if not matches:
        raise FileNotFoundError(f"Missing dataset file: {p} and no '(N)' variant found")

    matches.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    print(f"[warn] Using variant for missing {filename}: {matches[0].name}", flush=True)
    return matches[0]


def escape_nonascii(s: str) -> str:
    """
    Re-escape non-ASCII chars into literal \\uXXXX sequences.

    This preserves the tokenization difference between:
      - raw Unicode chars like • × ²
      - literal ASCII escape sequences like \\u2022
    """
    out = []
    for c in s:
        cp = ord(c)
        if cp < 128:
            out.append(c)
        elif cp <= 0xFFFF:
            out.append(f"\\u{cp:04x}")
        else:
            cp -= 0x10000
            hi = 0xD800 + (cp >> 10)
            lo = 0xDC00 + (cp & 0x3FF)
            out.append(f"\\u{hi:04x}\\u{lo:04x}")
    return "".join(out)


def load_jsonl(filename: str, limit: int, escape_ascii_variant: bool = False):
    data = []
    stats = {
        "rows_total": 0,
        "rows_kept": 0,
        "rows_nonascii_before": 0,
        "rows_nonascii_after": 0,
    }

    try:
        with open(filename, "r", encoding="utf-8") as f:
            for line in f:
                if len(data) >= limit:
                    break

                try:
                    item = json.loads(line)
                except Exception:
                    continue

                stats["rows_total"] += 1

                q = item.get("question") or item.get("prompt") or item.get("input") or ""
                a = item.get("response") or item.get("answer") or item.get("output") or ""

                if any(ord(c) >= 128 for c in q + a):
                    stats["rows_nonascii_before"] += 1

                if escape_ascii_variant:
                    q = escape_nonascii(q)
                    a = escape_nonascii(a)

                if any(ord(c) >= 128 for c in q + a):
                    stats["rows_nonascii_after"] += 1

                if q and a:
                    data.append({
                        "question": q,
                        "answer": a,
                        "document": f"Problem: {q}\nSolution: {a}",
                    })
                    stats["rows_kept"] += 1

    except FileNotFoundError:
        pass

    return data, stats


class ModelWrapper:
    """
    Computes answer-only average NLL:
    - split on "Solution:"
    - keep answer tokens even if long
    - trim prompt first
    - if answer alone too long, keep answer tail
    - mask prompt tokens with -100
    """
    def __init__(self, model, tokenizer, max_length: int, debug: bool = False, debug_max_print: int = 20):
        self.model = model
        self.tokenizer = tokenizer
        self.device = get_input_device(model)
        self.max_length = max_length
        self.debug = debug
        self.debug_max_print = debug_max_print
        self._dbg_prints = 0

    def get_loss_and_metadata(self, document: str):
        marker = "Solution:"
        idx = document.find(marker)
        if idx == -1:
            return None

        prompt_text = document[: idx + len(marker)]
        answer_text = document[idx + len(marker):].lstrip()
        if len(answer_text.strip()) == 0:
            return None

        prompt_ids = self.tokenizer(
            prompt_text,
            add_special_tokens=True,
            truncation=False,
        )["input_ids"]

        answer_ids = self.tokenizer(
            answer_text,
            add_special_tokens=False,
            truncation=False,
        )["input_ids"]

        if len(answer_ids) == 0:
            return None

        orig_prompt_len = len(prompt_ids)
        orig_answer_len = len(answer_ids)
        orig_total = orig_prompt_len + orig_answer_len

        if orig_total > self.max_length:
            if len(answer_ids) >= self.max_length:
                # Keep the tail of the answer, retain one prompt token.
                answer_ids = answer_ids[-(self.max_length - 1):]
                prompt_ids = prompt_ids[:1]
            else:
                # Trim prompt first.
                keep_prompt = self.max_length - len(answer_ids)
                prompt_ids = prompt_ids[-keep_prompt:]

        input_ids_list = prompt_ids + answer_ids
        labels_list = ([-100] * len(prompt_ids)) + answer_ids

        if self.debug and self._dbg_prints < self.debug_max_print:
            print(
                f"[DBG] prompt={len(prompt_ids)} answer={len(answer_ids)} total={len(input_ids_list)} "
                f"(orig_prompt={orig_prompt_len} orig_answer={orig_answer_len} orig_total={orig_total}) "
                f"trimmed={orig_total > self.max_length}",
                flush=True,
            )
            self._dbg_prints += 1

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

        loss = out.loss
        if loss is None or not torch.isfinite(loss):
            return None

        return {
            "loss": float(loss.detach().cpu()),
            "prompt_text": prompt_text,
            "answer_text": answer_text,
            "orig_prompt_tokens": orig_prompt_len,
            "orig_answer_tokens": orig_answer_len,
            "orig_total_tokens": orig_total,
            "used_prompt_tokens": len(prompt_ids),
            "used_answer_tokens": len(answer_ids),
            "used_total_tokens": len(input_ids_list),
            "trimmed": bool(orig_total > self.max_length),
        }


def main():
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("TRANSFORMERS_NUM_WORKERS_MATERIALIZE", "1")
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

    os.makedirs(OUT_DIR, exist_ok=True)

    script_dir = Path(__file__).resolve().parent
    datasets_dir = Path(DATASETS_DIR).expanduser() if DATASETS_DIR else (script_dir / "O1_UnicodeandASCII")

    torch_dtype = dtype_from_string(DTYPE)
    max_memory = build_max_memory()

    print("[info] model         =", MODEL_NAME, flush=True)
    print("[info] datasets_dir  =", str(datasets_dir), flush=True)
    print("[info] out_dir       =", OUT_DIR, flush=True)
    print("[info] max_length    =", MAX_LENGTH, flush=True)
    print("[info] limit         =", LIMIT_PER_DATASET, flush=True)
    print("[info] dtype         =", DTYPE, flush=True)
    print("[info] device_map    =", DEVICE_MAP, flush=True)
    print("[info] load_in_8bit  =", LOAD_IN_8BIT, flush=True)
    print("[info] max_memory    =", max_memory, flush=True)

    if torch.cuda.is_available():
        print("[info] cuda devices  =", torch.cuda.device_count(), flush=True)
        for i in range(torch.cuda.device_count()):
            _ = torch.zeros(1, device=f"cuda:{i}")
            print(f"[info] gpu[{i}] = {torch.cuda.get_device_name(i)}", flush=True)

    datasets = {}
    dataset_stats = {}

    print("\n--- Loading Datasets ---", flush=True)
    for label, spec in FILES_MAP.items():
        filename = spec["file"]
        transform = spec.get("transform")
        escape_flag = transform == "escape"

        print(f"[TRY] {label} -> {filename}", flush=True)
        try:
            resolved = resolve_dataset_file(datasets_dir, filename)
            print(f"[FOUND] {label} -> {resolved}", flush=True)
        except FileNotFoundError as e:
            print(f"[MISS] {label}: {e}", flush=True)
            continue

        loaded, stats = load_jsonl(
            str(resolved),
            limit=LIMIT_PER_DATASET,
            escape_ascii_variant=escape_flag,
        )

        print(f"[COUNT] {label}: {len(loaded)}", flush=True)
        print(
            f"        nonascii rows {stats['rows_nonascii_before']} -> {stats['rows_nonascii_after']}",
            flush=True,
        )

        if escape_flag and stats["rows_nonascii_after"] > 0:
            raise RuntimeError(f"ASCII transform failed for {label}: non-ASCII remains")

        if loaded:
            datasets[label] = loaded
            dataset_stats[label] = {
                "file": str(resolved),
                "transform": transform,
                **stats,
            }
            print(f"  ✅ {label}: {len(loaded)}", flush=True)
        else:
            print(f"  ❌ {label}: found file but parsed 0 usable rows", flush=True)

    if not datasets:
        raise RuntimeError("No datasets loaded.")

    model = None
    tok = None
    wrap = None

    try:
        print("\n--- Loading Tokenizer ---", flush=True)

        try:
            tok = AutoTokenizer.from_pretrained(
                MODEL_NAME,
                trust_remote_code=TRUST_REMOTE_CODE,
                local_files_only=LOCAL_FILES_ONLY,
                fix_mistral_regex=True,
            )
        except Exception:
            tok = AutoTokenizer.from_pretrained(
                MODEL_NAME,
                trust_remote_code=TRUST_REMOTE_CODE,
                local_files_only=LOCAL_FILES_ONLY,
            )

        tok.truncation_side = "left"
        tok.padding_side = "left"
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token

        print("\n--- Loading Model ---", flush=True)

        if LOAD_IN_8BIT:
            quant_config = BitsAndBytesConfig(load_in_8bit=True)

            # Transformers v5 prefers dtype=, older v4 used torch_dtype=.
            # Since this script is for your newer R1 env, use dtype= here.
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME,
                device_map=DEVICE_MAP,
                quantization_config=quant_config,
                dtype=torch_dtype,
                trust_remote_code=TRUST_REMOTE_CODE,
                local_files_only=LOCAL_FILES_ONLY,
                low_cpu_mem_usage=True,
                max_memory=max_memory,
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME,
                device_map=DEVICE_MAP,
                dtype=torch_dtype,
                trust_remote_code=TRUST_REMOTE_CODE,
                local_files_only=LOCAL_FILES_ONLY,
                low_cpu_mem_usage=True,
                max_memory=max_memory,
            )

        model.config.use_cache = False
        model.eval()

        model_max_length = min(MAX_LENGTH, get_model_ctx_limit(model, tok))
        wrap = ModelWrapper(model, tok, max_length=model_max_length, debug=DEBUG)
        print(f"[INFO] effective max_length = {model_max_length}", flush=True)

        output = {
            "model_name": MODEL_NAME,
            "datasets_dir": str(datasets_dir),
            "max_length": model_max_length,
            "max_length_requested": MAX_LENGTH,
            "limit_per_dataset": LIMIT_PER_DATASET,
            "dtype": DTYPE,
            "device_map": DEVICE_MAP,
            "load_in_8bit": LOAD_IN_8BIT,
            "max_memory": max_memory,
            "files_map": FILES_MAP,
            "dataset_stats": dataset_stats,
            "results": {},
        }

        print("\n--- Computing Losses ---", flush=True)
        for ds_label, samples in datasets.items():
            print(f"\n[dataset] {ds_label}", flush=True)

            rows = []
            valid_losses = []

            for i, sample in enumerate(samples):
                out = wrap.get_loss_and_metadata(sample["document"])

                row = {
                    "idx": i,
                    "question": sample["question"],
                    "answer": sample["answer"],
                    "loss": None,
                    "ok": False,
                }

                if out is not None:
                    row.update(out)
                    row["ok"] = True
                    valid_losses.append(out["loss"])

                rows.append(row)

                if (i + 1) % 10 == 0:
                    print(f"   done {i+1}/{len(samples)}", flush=True)

            summary = {
                "count": len(rows),
                "valid_count": len(valid_losses),
                "mean_loss": float(np.mean(valid_losses)) if valid_losses else None,
                "std_loss": float(np.std(valid_losses)) if valid_losses else None,
            }

            output["results"][ds_label] = {
                "summary": summary,
                "rows": rows,
            }

            print(f"   valid: {summary['valid_count']}/{summary['count']}", flush=True)
            print(f"   mean loss: {summary['mean_loss']}", flush=True)

        model_name_short = MODEL_NAME.split("/")[-1]
        out_path = Path(OUT_DIR) / f"{safe_name(model_name_short)}__o1_losses.json"

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

        print(f"\n✅ Saved losses to: {out_path}", flush=True)

    finally:
        if model is not None:
            del model
        if tok is not None:
            del tok
        if wrap is not None:
            del wrap
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\n✅ DONE.", flush=True)


if __name__ == "__main__":
    main()