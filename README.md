# GPT-Style Language Model in PyTorch

An implementation of a decoder-only Transformer, from tokenization and causal attention to next-token training and autoregressive generation.

This is my independent study implementation based on Sebastian Raschka's *Build a Large Language Model (From Scratch)*. It focuses on understanding the model internals and is not the book's official repository.

**Stack:** Python 3.11+ · PyTorch · tiktoken · NumPy · Matplotlib

## Implemented components

- Simple tokenizers and GPT-2 byte-pair encoding through `tiktoken`.
- Sliding-window datasets for next-token prediction.
- Self-attention, causal masking, and multi-head attention.
- Token and positional embeddings, pre-normalized Transformer blocks, residual connections, GELU feed-forward layers, and a language-model head.
- Training and validation loss calculation, an AdamW training loop, and text-generation examples.

## Repository layout

| Path | Contents |
| --- | --- |
| `src/llm_study/attention/` | Reusable attention implementations |
| `src/llm_study/tokenization/` | Simple tokenizer implementations |
| `src/llm_study/data/` | GPT dataset and data-loader helpers |
| `src/llm_study/model/` | GPT model and Transformer components |
| `notebooks/` | Topic-based runnable Python experiments |
| `data/` | Local dataset folders and data notes |
| `checkpoints/`, `outputs/` | Local model weights and experiment outputs |

## Setup

From the repository root:

```bash
python -m venv .venv
```

Activate the environment:

```bash
# Linux / macOS
source .venv/bin/activate
```

```powershell
# Windows PowerShell
.venv\Scripts\Activate.ps1
```

Install the package:

```bash
python -m pip install --upgrade pip
python -m pip install -e .
```

## Explore the implementation

Inspect the GPT architecture, output dimensions, and parameter counts without training:

```bash
python notebooks/gpt_model/model_inspection.py
```

This script instantiates a GPT-2-sized configuration with random weights; it is not a pretrained model. It may require substantial RAM.

The `notebooks/text_data/`, `notebooks/attention/`, and `notebooks/gpt_model/` directories contain smaller examples for individual components.

## Training

The training script reads `data/raw/the-verdict.txt`; provide that file before running it. Review the model configuration and training settings first:

```bash
python notebooks/pretraining/model_training.py
```

The script performs a train/validation split, evaluates cross-entropy loss, and generates sample text during training. Training can be slow on a CPU.

Local datasets, checkpoints, and generated outputs are ignored by Git. This repository presents the implementation and experiments; it does not claim pretrained-model quality or benchmark results.

## Reference

Sebastian Raschka, *Build a Large Language Model (From Scratch)*. The book supplies the learning sequence and architectural reference for this implementation.
