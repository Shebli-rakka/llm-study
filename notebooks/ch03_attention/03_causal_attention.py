import torch

from llm_from_scratch.attention.selfAttention import selfAttention_v2

inputs = torch.tensor(
    [
        [0.43, 0.15, 0.89],  # Your (x^1)
        [0.55, 0.87, 0.66],  # journey (x^2)
        [0.57, 0.85, 0.64],  # starts (x^3)
        [0.22, 0.58, 0.33],  # with (x^4)
        [0.77, 0.25, 0.10],  # one (x^5)
        [0.05, 0.80, 0.55],  # step (x^6)
    ]
)

d_in = inputs.shape[-1]
d_out = 2

# Calculating attention scores.
sa_v2 = selfAttention_v2(d_in, d_out)
queries = sa_v2.W_query(inputs)
keys = sa_v2.W_key(inputs)

attn_scores = queries @ keys.T
attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
print(attn_weights)


# Masking attention scores.
context_len = attn_weights.shape[0]
mask_simple = torch.tril(torch.ones(context_len, context_len))
print(mask_simple)

masked_attn_weights = attn_weights * mask_simple
print(masked_attn_weights)


# Normalize masked attention weights.
row_sums = masked_attn_weights.sum(dim=-1, keepdim=True)
print(row_sums)

masked_attn_weights_norm = masked_attn_weights / row_sums
print(masked_attn_weights_norm)


# Easier way: mask future positions before applying softmax.
mask = torch.triu(torch.ones(context_len, context_len), diagonal=1)
print(mask.bool())
masked_attn_scores = attn_scores.masked_fill(mask.bool(), -torch.inf)
print(masked_attn_scores)
masked_attn_weights_norm_2 = torch.softmax(
    masked_attn_scores / keys.shape[-1]**0.5,
    dim=-1,
)
print(masked_attn_weights_norm_2)


# Apply dropout to attention weights.
torch.manual_seed(123)
dropout = torch.nn.Dropout(0.5)
example = torch.ones(6, 6)
print(dropout(example))

print(dropout(masked_attn_weights_norm_2))
