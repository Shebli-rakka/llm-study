import torch
import tiktoken

from llm_from_scratch.GPT.DummyGPT import GPTModel
from llm_from_scratch.loaders.dataLoader import DATALoaderV1

GPT_CONFIG_124M = {
    "vocab_size": 50257,      # Vocabulary size
    "context_length": 256,    # Context length
    "emb_dim": 768,           # Embedding dimension
    "n_heads": 12,            # Number of attention heads
    "n_layers": 12,           # Number of layers
    "drop_rate": 0.1,         # Dropout rate
    "qkv_bias": False,        # Query-Key-Value bias
}

torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)
model.eval()


def generate_text_simple(model, idx, max_new_tokens, context_size):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)

        logits = logits[:, -1, :]
        probas = torch.softmax(logits, dim=-1)
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)
        idx = torch.cat((idx, idx_next), dim=1)

    return idx


generate_text_simpe = generate_text_simple


def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    return torch.tensor(encoded).unsqueeze(0)


def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)
    return tokenizer.decode(flat.tolist())


start_context = "Every effort moves you"
tokenizer = tiktoken.get_encoding("gpt2")

token_ids = generate_text_simple(
    model=model,
    idx=text_to_token_ids(start_context, tokenizer),
    max_new_tokens=10,
    context_size=GPT_CONFIG_124M["context_length"],
)

print("Output text:\n", token_ids_to_text(token_ids, tokenizer))


inputs = torch.tensor(
    [
        [16833, 3626, 6100],
        [40, 1107, 588],
    ]
)

targets = torch.tensor(
    [
        [3626, 6100, 345],
        [1107, 588, 11311],
    ]
)

with torch.no_grad():
    logits = model(inputs)

probas = torch.softmax(logits, dim=-1)
print(probas.shape)

next_tokens_ids = torch.argmax(probas, dim=-1, keepdim=True)
print("Token IDs:\n", next_tokens_ids)

print(f"Targets batch 1: {token_ids_to_text(targets[0], tokenizer)}")
print(f"Outputs batch 1:"
      f" {token_ids_to_text(next_tokens_ids[0].flatten(), tokenizer)}")

print(f"Targets batch 2: {token_ids_to_text(targets[1], tokenizer)}")
print(f"Outputs batch 2:"
      f" {token_ids_to_text(next_tokens_ids[1].flatten(), tokenizer)}")


txt_idx = 0
target_probas_1 = probas[txt_idx, [0, 1, 2], targets[txt_idx]]
print("Text 1:", target_probas_1)

txt_idx = 1
target_probas_2 = probas[txt_idx, [0, 1, 2], targets[txt_idx]]
print("Text 2:", target_probas_2)

log_probas = torch.log(torch.cat((target_probas_1, target_probas_2)))
print(log_probas)

avg_log_probas = torch.mean(log_probas)
print(avg_log_probas)

neg_avg_log_probas = -1 * avg_log_probas
print(neg_avg_log_probas)

#cross entropy loss
print("Logits shape:", logits.shape)
print("Targets shape:", targets.shape)

logits_flat = logits.flatten(0, 1)
targets_flat = targets.flatten()
print("Flattened logits:", logits_flat.shape)
print("Flattened targets:", targets_flat.shape)

loss = torch.nn.functional.cross_entropy(logits_flat, targets_flat)
print(loss)

perplexity = torch.exp(loss)
print(perplexity)
# In this example, perplexity means the model is roughly choosing among
# this many possible next tokens.


# Calculating training and validation losses.
with open(r"data/raw/the-verdict.txt", "r", encoding="utf-8") as f:
    text_data = f.read()



total_characters = len(text_data)
total_tokens = len(tokenizer.encode(text_data))
print("Characters:", total_characters)
print("Tokens:", total_tokens)

train_ratio = 0.90
split_idx = int(train_ratio * len(text_data))
train_data = text_data[:split_idx]
val_data = text_data[split_idx:]

torch.manual_seed(123)

train_loader = DATALoaderV1().create_data_loader(train_data,
                                                 batch_size=2,
                                                 max_length=GPT_CONFIG_124M["context_length"],
                                                 stride=GPT_CONFIG_124M["context_length"],
                                                 shuffle=True,
                                                 drop_last=True,
                                                 num_workers=0)

val_loader = DATALoaderV1().create_data_loader(val_data, batch_size = 2,
                                               max_length=GPT_CONFIG_124M["context_length"],
                                               stride=GPT_CONFIG_124M["context_length"],
                                               shuffle=True,
                                               drop_last=True,
                                               num_workers=0)

print("Train Loader:")
for x, y in train_loader:
    print(x.shape, y.shape)

print("\nVal Loader:")
for x, y in val_loader:
    print(x.shape, y.shape)


def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(input_batch)
    return torch.nn.functional.cross_entropy(
        logits.flatten(0, 1),
        target_batch.flatten(),
    )


def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))

    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss
        else:
            break
    return total_loss / num_batches


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
with torch.no_grad():
    train_loss = calc_loss_loader(train_loader, model, device)
    val_loss = calc_loss_loader(val_loader, model, device)
print("Training loss:", train_loss)
print("Validation loss:", val_loss)
