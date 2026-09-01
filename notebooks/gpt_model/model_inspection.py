import torch
import tiktoken

from llm_study.model.gpt import GPTModel

GPT_CONFIG_124M = {
    "vocab_size": 50257,      # Vocabulary size
    "context_length": 1024,   # Context length
    "emb_dim": 768,           # Embedding dimension
    "n_heads": 12,            # Number of attention heads
    "n_layers": 12,           # Number of layers
    "drop_rate": 0.1,         # Dropout rate
    "qkv_bias": False,        # Query-Key-Value bias
}

tokenizer = tiktoken.get_encoding("gpt2")
batch = []
txt1 = "Every effort moves you"
txt2 = "Every day holds a"

batch.append(torch.tensor(tokenizer.encode(txt1)))
batch.append(torch.tensor(tokenizer.encode(txt2)))
batch = torch.stack(batch, dim=0)


torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)
out = model(batch)
print("Input batch:\n", batch)
print("\nOutput shape:", out.shape)
print(out)

total_params = sum(p.numel() for p in model.parameters())
print(f"Total number of parameters: {total_params:,}")

# Weight tying was used in the original GPT-2 architecture:
# the output layer reuses the token embedding weights.

total_params_gpt2 = total_params - sum(p.numel() for p in model.out_head.parameters())
print(f"Number of trainable parameters "
      f"considering weight tying: {total_params_gpt2:,}"
)

total_size_bytes = total_params * 4
total_size_mb = total_size_bytes / (1024 * 1024)
print(f"Total size of the model: {total_size_mb:.2f} MB")

total_params_mha = sum(
    p.numel()
    for trf in model.trf_blocks
    for p in trf.att.parameters()
)

print(f"Total number of parameters in attention modules: {total_params_mha:,}")

total_params_ff = sum(
    p.numel()
    for trf in model.trf_blocks
    for p in trf.ff.parameters()
)

print(f"Total number of parameters in feed forward modules: {total_params_ff:,}")
