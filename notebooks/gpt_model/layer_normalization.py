import torch
import torch.nn as nn

from llm_study.model.gpt import LayerNorm


batch_example = torch.rand(2, 5)
layer = nn.Sequential(nn.Linear(5, 6), nn.ReLU())
out = layer(batch_example)
print(out)

mean = out.mean(dim=-1, keepdim=True)
var = out.var(dim=-1, keepdim=True)
print("mean:\n", mean)
print("var:\n", var)
norm_out = (out - mean) / torch.sqrt(var)
mean = norm_out.mean(dim=-1, keepdim=True)
var = norm_out.var(dim=-1, keepdim=True)
print(norm_out)
print("mean:\n", mean)
print("var:\n", var)

torch.set_printoptions(sci_mode=False)
print("mean:\n", mean)
print("var:\n", var)

# Compare manual normalization with the LayerNorm implementation.
out = layer(batch_example)
mean = out.mean(dim=-1, keepdim=True)
var = out.var(dim=-1, keepdim=True)
print("mean:\n", mean)
print("var:\n", var)

ln = LayerNorm(6)
norm_out = ln(out)
mean = norm_out.mean(dim=-1, keepdim=True)
var = norm_out.var(dim=-1, keepdim=True, unbiased=False)
print("mean:\n", mean)
print("var:\n", var)
