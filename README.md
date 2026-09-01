# llm-study

This repository is my personal implementation workspace based on Sebastian Raschka's book *Build a Large Language Model (From Scratch)*.

The goal is to understand and implement the mechanisms myself rather than copy the official repository or use a high-level LLM framework.

This is a personal study project built while following the book. The code is organized by topic and keeps the implementation close to the learning process.

## Repository layout

- `notebooks/` contains topic-based experiments and runnable scripts.
- `src/llm_study/` contains reusable code promoted from the experiments.
- `src/llm_study/attention/` contains self-attention, causal attention, and multi-head attention implementations.
- `src/llm_study/tokenization/` contains the simple tokenizer implementations.
- `src/llm_study/data/` contains the GPT dataset and data loader helpers.
- `src/llm_study/model/` contains the GPT model, transformer block, layer norm, GELU, and feed-forward modules.
- `checkpoints/` contains locally saved model weights.
- `outputs/` contains generated text, metrics, and other experiment outputs.

## Environment setup

Create and activate a Python 3.11 environment, then install the project in editable mode:

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e .
```

## Notes

- This project is an independent study implementation based on the book, not the official source repository.
- The repository includes my own code and notes written while studying the material.
- Local datasets, checkpoints, and generated outputs are ignored by Git by default.
