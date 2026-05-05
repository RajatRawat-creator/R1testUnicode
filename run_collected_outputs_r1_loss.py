#!/usr/bin/env python3
# Single-model loss cache script for DeepSeek-R1 across collected_outputs.
# - Same R1 loading + answer-only NLL machinery as run_o1_loss_r1.py.
# - No argparse: edit settings below, then run `python run_r1_loss_collected.py`.

import gc
import json
import os
import re
from pathlib import Path

# ============================================================
# Disable Transformers' hub-kernel mapping BEFORE importing
# transformers. The kernels-community/deep-gemm prebuilt wheel
# can throw `deep_gemm::DGException` at runtime on GPU/CUDA
# combos it wasn't tuned for; the built-in Triton/PyTorch FP8
# fallback is more portable.
# ============================================================
os.environ.setdefault("DISABLE_KERNEL_MAPPING", "1")
os.environ.setdefault("HF_HUB_DISABLE_KERNELS", "1")
# Uncomment to localize any future CUDA assert to the real op:
# os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")

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


# ============================================================
# Compatibility patch for transformers v5 FineGrainedFP8 +
# DeepseekV3Config. The library's _first_attr() helper only
# probes ("num_local_experts", "num_experts") but DeepSeek-V3
# stores the count as `n_routed_experts`, and the new dataclass
# config doesn't go through PretrainedConfig.__getattribute__'s
# attribute_map remap, so hasattr() returns False and load fails.
# ============================================================
try:
    import transformers.integrations.finegrained_fp8 as _tf_fp8

    _orig_first_attr = _tf_fp8._first_attr

    def _patched_first_attr(obj, *names):
        extra = ("n_routed_experts",)
        for name in (*names, *extra):
            if hasattr(obj, name):
                return getattr(obj, name)
        raise AttributeError(
            f"{type(obj).__name__} has none of: {(*names, *extra)}"
        )

    _tf_fp8._first_attr = _patched_first_attr
except Exception as _e:  # pragma: no cover
    print(f"[WARN] could not patch finegrained_fp8._first_attr: {_e}", flush=True)


# ============================================================
# Compatibility patch for transformers v5 FineGrainedFP8:
# replace_with_fp8_linear() mutates the module tree while
# iterating model.named_modules(). After replacing `...experts`
# (a ModuleList) with a flat FP8Experts, the generator keeps
# walking the OLD ModuleList's children (experts.0.gate_proj,
# ...) and then calls set_submodule("...experts.0.gate_proj",
# FP8Linear), which fails with:
#   AttributeError: FP8Experts has no attribute `0`
# We monkey-patch to (1) snapshot the module list first and
# (2) skip any name that lives under an already-replaced prefix.
# ============================================================
try:
    import torch.nn as _nn
    import transformers.integrations.finegrained_fp8 as _tf_fp8m

    _orig_replace_with_fp8_linear = _tf_fp8m.replace_with_fp8_linear

    def _patched_replace_with_fp8_linear(
        model,
        modules_to_not_convert=None,
        quantization_config=None,
        pre_quantized=False,
    ):
        if quantization_config is None or getattr(
            quantization_config, "dequantize", False
        ):
            import inspect as _inspect

            _orig_params = _inspect.signature(
                _orig_replace_with_fp8_linear
            ).parameters
            _kwargs = {
                "modules_to_not_convert": modules_to_not_convert,
                "quantization_config": quantization_config,
            }
            if "pre_quantized" in _orig_params:
                _kwargs["pre_quantized"] = pre_quantized
            return _orig_replace_with_fp8_linear(model, **_kwargs)

        # Older transformers (<5) don't expose FP8Experts /
        # use_experts_implementation; the experts-vs-ModuleList
        # bug only exists in v5, so just defer to the original.
        if not all(
            hasattr(_tf_fp8m, _n)
            for _n in (
                "FP8Experts",
                "ALL_FP8_EXPERTS_FUNCTIONS",
                "use_experts_implementation",
                "should_convert_module",
            )
        ):
            import inspect as _inspect

            _orig_params = _inspect.signature(
                _orig_replace_with_fp8_linear
            ).parameters
            _kwargs = {
                "modules_to_not_convert": modules_to_not_convert,
                "quantization_config": quantization_config,
            }
            if "pre_quantized" in _orig_params:
                _kwargs["pre_quantized"] = pre_quantized
            return _orig_replace_with_fp8_linear(model, **_kwargs)

        FP8Linear = _tf_fp8m.FP8Linear
        FP8Experts = _tf_fp8m.FP8Experts
        ALL_FP8_EXPERTS_FUNCTIONS = _tf_fp8m.ALL_FP8_EXPERTS_FUNCTIONS
        use_experts_implementation = _tf_fp8m.use_experts_implementation
        should_convert_module = _tf_fp8m.should_convert_module

        # Snapshot the tree first.
        snapshot = list(model.named_modules())
        pending = []
        replaced_prefixes = []

        for module_name, module in snapshot:
            if not should_convert_module(module_name, modules_to_not_convert):
                continue
            # Skip anything inside an already-scheduled replacement.
            if any(
                module_name == p or module_name.startswith(p + ".")
                for p in replaced_prefixes
            ):
                continue

            module_kwargs = {} if pre_quantized else {"dtype": None}
            new_module = None
            with torch.device("meta"):
                if module_name.endswith(".experts"):
                    has_gate = getattr(module, "has_gate", True)
                    has_bias = getattr(module, "has_bias", False)
                    config = getattr(
                        module, "config", model.config.get_text_config()
                    )
                    new_class = use_experts_implementation(
                        experts_class=FP8Experts,
                        experts_interface=ALL_FP8_EXPERTS_FUNCTIONS,
                        has_bias=has_bias,
                        has_gate=has_gate,
                    )
                    new_module = new_class(
                        config=config,
                        block_size=quantization_config.weight_block_size,
                        activation_scheme=quantization_config.activation_scheme,
                        has_bias=has_bias,
                        has_gate=has_gate,
                        **module_kwargs,
                    )
                    replaced_prefixes.append(module_name)
                elif isinstance(module, _nn.Linear):
                    new_module = FP8Linear(
                        in_features=module.in_features,
                        out_features=module.out_features,
                        block_size=quantization_config.weight_block_size,
                        activation_scheme=quantization_config.activation_scheme,
                        has_bias=module.bias is not None,
                        **module_kwargs,
                    )
            if new_module is not None:
                pending.append((module_name, new_module))

        for name, new_module in pending:
            model.set_submodule(name, new_module)

        if not pending:
            import logging

            logging.getLogger(__name__).warning(
                "fp8 patch: no linear/experts modules were replaced."
            )
        return model

    _tf_fp8m.replace_with_fp8_linear = _patched_replace_with_fp8_linear
    # The quantizer imported the symbol at module load time, repoint it too.
    try:
        import transformers.quantizers.quantizer_finegrained_fp8 as _tf_q_fp8

        _tf_q_fp8.replace_with_fp8_linear = _patched_replace_with_fp8_linear
    except Exception:
        pass
except Exception as _e:  # pragma: no cover
    print(
        f"[WARN] could not patch finegrained_fp8.replace_with_fp8_linear: {_e}",
        flush=True,
    )


# ============================================================
# Compatibility patch for FP8 weight initialization.
#
# DeepSeek-R1's checkpoint stores Linear weights as
# torch.float8_e4m3fn. After loading, transformers v5 calls
# _initialize_missing_keys -> initialize_weights -> smart_apply,
# which invokes the remote modeling_deepseek.py's _init_weights:
#
#     module.weight.data.normal_(mean=0.0, std=std)
#
# This crashes with:
#     RuntimeError: "normal_kernel_cuda" not implemented for 'Float8_e4m3fn'
#
# The FP8 weights were already loaded from the checkpoint and
# must NOT be reinitialized anyway. Make Tensor.normal_ a no-op
# for FP8 dtypes so the unconditional re-init silently skips them.
# ============================================================
try:
    _FP8_DTYPES = tuple(
        dt for dt in (
            getattr(torch, "float8_e4m3fn", None),
            getattr(torch, "float8_e5m2", None),
            getattr(torch, "float8_e4m3fnuz", None),
            getattr(torch, "float8_e5m2fnuz", None),
        )
        if dt is not None
    )

    _orig_tensor_normal_ = torch.Tensor.normal_

    def _safe_tensor_normal_(self, *args, **kwargs):
        if self.dtype in _FP8_DTYPES:
            # Loaded FP8 weights; do not touch.
            return self
        return _orig_tensor_normal_(self, *args, **kwargs)

    torch.Tensor.normal_ = _safe_tensor_normal_
except Exception as _e:  # pragma: no cover
    print(f"[WARN] could not patch torch.Tensor.normal_ for FP8: {_e}", flush=True)


# ============================================================
# Compatibility patch for transformers v5 finegrained-fp8
# Triton kernel loader.
#
# transformers calls
#     hub_kernels.lazy_load_kernel("finegrained-fp8")
# which under the hood does
#     get_kernel("kernels-community/finegrained-fp8", version=1)
# That call can silently fall back to None (FileNotFoundError /
# AssertionError swallowed in lazy_load_kernel) when the pinned
# hub version doesn't have a wheel matching the local torch+cuda
# variant. The downstream _load_triton_kernel then sees all four
# expected attributes as None and raises:
#     ImportError: finegrained-fp8 kernel is missing required
#     functions: w8a8_fp8_matmul, fp8_act_quant,
#     w8a8_fp8_matmul_batched, w8a8_fp8_matmul_grouped.
#
# We replace _load_triton_kernel with a version that fetches the
# kernel via the unpinned `get_kernel` path (which we verified
# does expose all four functions for the current torch build).
# ============================================================
try:
    import transformers.integrations.finegrained_fp8 as _tf_fp8k
    from transformers.integrations.hub_kernels import get_kernel as _tf_get_kernel

    def _patched_load_triton_kernel():
        if _tf_fp8k._triton_available is True:
            return
        if _tf_fp8k._triton_available is False:
            # Re-attempt anyway: previous failure was due to the
            # version-pinned lazy loader, not a real load failure.
            _tf_fp8k._triton_available = None

        _tf_fp8k._triton_available = False  # mark attempted

        kernel = _tf_get_kernel("kernels-community/finegrained-fp8")

        _tf_fp8k.triton_fp8_matmul = getattr(kernel, "w8a8_fp8_matmul", None)
        _tf_fp8k.triton_fp8_act_quant = getattr(kernel, "fp8_act_quant", None)
        _tf_fp8k.triton_batched_fp8_matmul = getattr(
            kernel, "w8a8_fp8_matmul_batched", None
        )
        _tf_fp8k.triton_grouped_fp8_matmul = getattr(
            kernel, "w8a8_fp8_matmul_grouped", None
        )

        missing = [
            n for n, a in [
                ("w8a8_fp8_matmul", _tf_fp8k.triton_fp8_matmul),
                ("fp8_act_quant", _tf_fp8k.triton_fp8_act_quant),
                ("w8a8_fp8_matmul_batched", _tf_fp8k.triton_batched_fp8_matmul),
                ("w8a8_fp8_matmul_grouped", _tf_fp8k.triton_grouped_fp8_matmul),
            ] if a is None
        ]
        if missing:
            raise ImportError(
                "finegrained-fp8 kernel (patched loader) is still missing: "
                + ", ".join(missing)
            )

        _tf_fp8k._triton_available = True

    _tf_fp8k._load_triton_kernel = _patched_load_triton_kernel
except Exception as _e:  # pragma: no cover
    print(
        f"[WARN] could not patch finegrained_fp8._load_triton_kernel: {_e}",
        flush=True,
    )


# ============================================================
# Compatibility patch for torch 2.6 + finegrained-fp8 kernel.
#
# The hub kernel uses PEP 585 generic annotations like
#     block_size: list[int]
# inside @triton_op. torch 2.6's torch._library.infer_schema only
# whitelists `typing.List[int]`, so it raises:
#     ValueError: infer_schema(func): Parameter block_size has
#         unsupported type list[int].
#
# We extend SUPPORTED_PARAM_TYPES with the lowercase generic
# variants (list[T], tuple[T, ...]) mapped to the same schema
# strings as their typing.* counterparts.
# ============================================================
try:
    import typing as _typing
    import torch._library.infer_schema as _torch_infer_schema

    _SPT = _torch_infer_schema.SUPPORTED_PARAM_TYPES
    _new_entries = {}
    for _typ, _schema in list(_SPT.items()):
        # Translate typing.List[T] / typing.Sequence[T] -> list[T]
        _origin = getattr(_typ, "__origin__", None)
        _args = getattr(_typ, "__args__", None)
        if _origin in (list, _typing.List) and _args:
            _lower = list[_args[0]]  # e.g. list[int]
            if _lower not in _SPT:
                _new_entries[_lower] = _schema
        # Optional[list/Sequence[T]]
        if _origin is _typing.Union and _args:
            _non_none = [a for a in _args if a is not type(None)]
            if len(_non_none) == 1:
                _inner = _non_none[0]
                _inner_origin = getattr(_inner, "__origin__", None)
                _inner_args = getattr(_inner, "__args__", None)
                if _inner_origin in (list, _typing.List) and _inner_args:
                    _lower = _typing.Optional[list[_inner_args[0]]]
                    if _lower not in _SPT:
                        _new_entries[_lower] = _schema
    _SPT.update(_new_entries)
except Exception as _e:  # pragma: no cover
    print(
        f"[WARN] could not patch torch infer_schema for PEP 585 generics: {_e}",
        flush=True,
    )


# =========================
# HARD-CODED SETTINGS
# =========================

MODEL_NAME = "deepseek-ai/DeepSeek-R1"
DATASETS_DIR = "collected_outputs"
OUT_DIR = "./outputs_r1_collected"

# For smoke test, use 2.
# For real run, change to 200.
LIMIT_PER_DATASET = 200

MAX_LENGTH = 32768

# Cap on answer tokens (post-tokenize, pre max_length trim).
# Set to None to disable. Keeps the head of the answer.
MAX_ANSWER_TOKENS = 2048

# Choose one of: "float16", "bfloat16", "float32"
DTYPE = "bfloat16"

# Usually keep as "auto"
DEVICE_MAP = "auto"

TRUST_REMOTE_CODE = True            # tokenizer may still need remote code
TRUST_REMOTE_CODE_MODEL = False     # native DeepseekV3ForCausalLM is FP8-aware;
                                    # remote modeling_deepseek.py's moe_infer
                                    # does len(self.experts)/self.experts[i],
                                    # which breaks against fused FP8Experts.
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
    "Gemma-3-27B-it (OMI, Response Only)": "gemma-3-27b-it__openmath_jsonl__chat__vllm (1).jsonl",
    "Gemma-3-27B-it (s1, Response Only)": "gemma-3-27b-it__s1k__chat__vllm.jsonl",

    "Claude-3.5-Sonnet (OMI, Response Only)": "Claude-Sonnet-3.5_omi(200).jsonl",
    "Claude-3.5-Sonnet (s1, Response Only)": "Claude-Sonnet-3.5_s1(200).jsonl",

    "DeepSeek R1 (OMI, Trace + Response)": "R1_omi(200)_reasoningonly.jsonl",
    "DeepSeek R1 (s1, Trace + Response)": "R1_s1(200)_reasoningonly.jsonl",

    "DeepSeek R1 (OMI, Response Only)": "R1_omi(200)_responseonly.jsonl",
    "DeepSeek R1 (s1, Response Only)": "R1_s1(200)_responseonly.jsonl",

    "Qwen-3-235B-A22B-Thinking-2507 (OMI, Trace + Response)": "Qwen-3-235B_omi(200)_reasoningonly.jsonl",
    "Qwen-3-235B-A22B-Thinking-2507 (s1, Trace + Response)": "Qwen-3-235B_s1(200)_reasoningonly.jsonl",

    "Claude Opus 4.5 (OMI, Trace + Response)": "claude-opus-4-5-20251101__openmath_jsonl__chat__claude_reasoning_only__REPAIRED.jsonl",
    "Claude Opus 4.5 (s1, Trace + Response)": "claude-opus-4-5-20251101__s1k__chat__claude_reasoning_only__REPAIRED.jsonl",

    "Claude Opus 4.6 (OMI, Trace + Response)": "Claude-Opus-4.6_omi(200)_reasoningonly.jsonl",
    "Claude Opus 4.6 (s1, Trace + Response)": "Claude-Opus-4.6_s1(200)_reasoningonly.jsonl",

    "o1 (OMI, Response Only)": "o1_omi(200).jsonl",
    "o1 (s1, Response Only)": "o1_s1(200).jsonl",

    "o3 (OMI, Response Only)": "o3_omi(200).jsonl",
    "o3 (s1, Response Only)": "o3_s1(200).jsonl",

    "Llama-3.3-70B-Instruct (OMI, Response)": "Llama-3.3-70B-Instruct_omi(200).jsonl",
    "Llama-3.3-70B-Instruct (s1, Response)": "Llama-3.3-70B-Instruct_s1(200).jsonl",

    "QwQ-32B (OMI, Trace + Response)": "QwQ-32B-omi(200).jsonl",
    "QwQ-32B (s1, Trace + Response)": "QwQ-32B-s1(200).jsonl",

    "QwQ-32B Preview (OMI, Trace + Response)": "QwQ-32B-Preview-omi(200).jsonl",
    "QwQ-32B Preview (s1, Trace + Response)": "QwQ-32B-Preview-s1(200).jsonl",
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

    # Recursive search by exact basename (collected_outputs may have subdirs).
    rglob_matches = [x for x in datasets_dir.rglob(filename) if x.is_file()]
    if rglob_matches:
        rglob_matches.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        if len(rglob_matches) > 1:
            print(
                f"[warn] Multiple matches for {filename}; using newest: {rglob_matches[0]}",
                flush=True,
            )
        return rglob_matches[0]

    stem = filename[:-6] if filename.endswith(".jsonl") else filename
    pattern = re.compile(rf"^{re.escape(stem)}(\s*\(\d+\))?\.jsonl$")
    matches = [x for x in datasets_dir.rglob("*.jsonl") if x.is_file() and pattern.match(x.name)]
    if not matches:
        raise FileNotFoundError(f"Missing dataset file: {p} and no '(N)' variant found")

    matches.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    print(f"[warn] Using variant for missing {filename}: {matches[0]}", flush=True)
    return matches[0]


def load_jsonl(filename: str, limit: int):
    data = []
    stats = {
        "rows_total": 0,
        "rows_kept": 0,
        "rows_nonascii": 0,
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
                    stats["rows_nonascii"] += 1

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
    def __init__(self, model, tokenizer, max_length: int, max_answer_tokens=None, debug: bool = False, debug_max_print: int = 20):
        self.model = model
        self.tokenizer = tokenizer
        self.device = get_input_device(model)
        self.max_length = max_length
        self.max_answer_tokens = max_answer_tokens
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

        # Cap answer to max_answer_tokens (keep head).
        if self.max_answer_tokens is not None and len(answer_ids) > self.max_answer_tokens:
            answer_ids = answer_ids[:self.max_answer_tokens]

        eff_total = len(prompt_ids) + len(answer_ids)

        if eff_total > self.max_length:
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
                f"(orig_prompt={orig_prompt_len} orig_answer={orig_answer_len} orig_total={orig_total} "
                f"eff_total={eff_total}) trimmed={eff_total > self.max_length}",
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
            "capped": bool(self.max_answer_tokens is not None and orig_answer_len > self.max_answer_tokens),
            "trimmed": bool(eff_total > self.max_length),
        }


def main():
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("TRANSFORMERS_NUM_WORKERS_MATERIALIZE", "1")
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

    os.makedirs(OUT_DIR, exist_ok=True)

    script_dir = Path(__file__).resolve().parent
    datasets_dir = Path(DATASETS_DIR).expanduser() if DATASETS_DIR else (script_dir / "collected_outputs")

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
    for label, filename in FILES_MAP.items():
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
        )

        print(f"[COUNT] {label}: {len(loaded)}", flush=True)
        print(f"        nonascii rows: {stats['rows_nonascii']}", flush=True)

        if loaded:
            datasets[label] = loaded
            dataset_stats[label] = {
                "file": str(resolved),
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
                trust_remote_code=TRUST_REMOTE_CODE_MODEL,
                local_files_only=LOCAL_FILES_ONLY,
                low_cpu_mem_usage=True,
                max_memory=max_memory,
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME,
                device_map=DEVICE_MAP,
                dtype=torch_dtype,
                trust_remote_code=TRUST_REMOTE_CODE_MODEL,
                local_files_only=LOCAL_FILES_ONLY,
                low_cpu_mem_usage=True,
                max_memory=max_memory,
            )

        model.config.use_cache = False
        model.eval()

        model_max_length = min(MAX_LENGTH, get_model_ctx_limit(model, tok))
        wrap = ModelWrapper(model, tok, max_length=model_max_length, max_answer_tokens=MAX_ANSWER_TOKENS, debug=DEBUG)
        print(f"[INFO] effective max_length = {model_max_length}", flush=True)
        print(f"[INFO] max_answer_tokens    = {MAX_ANSWER_TOKENS}", flush=True)

        output = {
            "model_name": MODEL_NAME,
            "datasets_dir": str(datasets_dir),
            "max_length": model_max_length,
            "max_length_requested": MAX_LENGTH,
            "max_answer_tokens": MAX_ANSWER_TOKENS,
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

            # Incremental save after each dataset; overwritten next iteration.
            inc_path = Path(OUT_DIR) / f"{safe_name(MODEL_NAME.split('/')[-1])}__collected_losses.json"
            with open(inc_path, "w", encoding="utf-8") as f:
                json.dump(output, f, indent=2, ensure_ascii=False)
            print(f"   [checkpoint] {inc_path}", flush=True)

        model_name_short = MODEL_NAME.split("/")[-1]
        out_path = Path(OUT_DIR) / f"{safe_name(model_name_short)}__collected_losses.json"

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
