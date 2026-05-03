# R1testUnicode

This repository contains a small loss-scoring experiment for comparing OpenAI `o1` response files under different JSON/text serialization formats, including default ASCII-escaped JSON and Unicode-preserving JSON.

The main script computes answer-only loss on saved `o1` responses using a Hugging Face causal language model.

## Repository contents

```text
R1testUnicode/
├── O1_UnicodeandASCII/
│   ├── o1__responses_default.jsonl
│   └── o1_openmath__responses_unicode.jsonl
├── run_o1_loss_r1.py
└── requirements.txt

## Setup

~~~bash
pip install -r requirements.txt
~~~

## Run

Run:

~~~bash
python run_o1_loss_r1.py
~~~

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

Install all via:

~~~bash
pip install -r requirements.txt
~~~
