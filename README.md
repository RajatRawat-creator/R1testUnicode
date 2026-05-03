# R1testUnicode

This repo contains:
- `run_o1_loss_r1.py` — computes answer-only loss for DeepSeek-R1 on saved OpenAI `o1` responses
- `O1_UnicodeandASCII/` — JSONL response files comparing Unicode-preserving vs ASCII/default serialization
- `requirements.txt`

## Setup

~~~bash
pip install -r requirements.txt
~~~

## Run

Run:

~~~bash
python run_o1_loss_r1.py
~~~

By default, the script scores up to 200 examples from each file and writes results to:

~~~text
outputs_r1/DeepSeek-R1__o1_losses.json
~~~

## Notes

- Uses `deepseek-ai/DeepSeek-R1` as the scoring model.
- Computes answer-only average negative log-likelihood.
- Compares:
  - `o1_openmath__responses_unicode.jsonl` — Unicode-preserving responses
  - `o1__responses_default.jsonl` — default/ASCII-style responses
- `LOAD_IN_8BIT = False` by default. Keep this setting unless 8-bit loading works in your environment.
- Uses `bfloat16` by default through `DTYPE = "bfloat16"`.
- The script is hard-coded: edit the settings at the top of `run_o1_loss_r1.py` if you want to change model, dataset directory, output directory, limit, dtype, or max length.

## Hugging Face Cache

For large models such as DeepSeek-R1, set cache to a large disk:

~~~bash
export HF_HOME=/data/<your_username>/hf_cache
export HF_HUB_CACHE=$HF_HOME/hub
export HF_DATASETS_CACHE=$HF_HOME/datasets
mkdir -p "$HF_HUB_CACHE" "$HF_DATASETS_CACHE"
~~~

Otherwise models may fill your home directory.

## Requirements

Main dependencies:
- torch
- transformers
- accelerate
- bitsandbytes
- numpy
- sentencepiece
- safetensors
- huggingface-hub
- tokenizers
- protobuf
- einops
- hf-xet

Install all via:

~~~bash
pip install -r requirements.txt
~~~
