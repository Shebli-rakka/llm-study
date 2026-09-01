import torch


# Input sequence (6 tokens).
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


# Compute attention scores w between x^2 and all other input elements.
query = inputs[1]
attn_scores_2 = torch.empty(inputs.shape[0])
for i, x_i in enumerate(inputs):
    attn_scores_2[i] = torch.dot(query, x_i)
print(attn_scores_2)

res = 0
for index, element in enumerate(inputs[0]):
    res += element * query[index]
print(res)


# Calculate attention weights by normalizing attention scores.
attn_weights_2 = attn_scores_2 / attn_scores_2.sum()
print(attn_weights_2)
print(attn_weights_2.sum())


# We often use the softmax function for normalization.
def softmax_naive(x):
    return torch.exp(x) / torch.exp(x).sum(dim=0)


attn_weights_2_naive = softmax_naive(attn_scores_2)
print(attn_weights_2_naive)
print(attn_weights_2_naive.sum())


# Softmax from PyTorch is more stable.
attn_weights_2 = torch.softmax(attn_scores_2, dim=0)
print(attn_weights_2)
print(attn_weights_2.sum(dim=0))


# Calculate the context vector for x^2.
context_vec_2 = torch.empty(query.shape[0])
for i, x_i in enumerate(inputs):
    context_vec_2 += attn_weights_2[i] * x_i
print(context_vec_2)


# Now compute attention weights for all input tokens.
attn_scores = torch.empty(6, 6)
for i, x_i in enumerate(inputs):
    for j, x_j in enumerate(inputs):
        attn_scores[i][j] = torch.dot(x_i, x_j)

print(attn_scores)


# More efficient matrix multiplication version.
attn_scores = inputs @ inputs.T
print(attn_scores)


# Calculate attention weights and context vectors.
attn_weights = torch.softmax(attn_scores, dim=-1)
print(attn_weights)
print(attn_weights.sum(dim=-1))

context_vecs = attn_weights @ inputs
print(context_vecs)
