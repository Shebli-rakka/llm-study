import tiktoken

from llm_from_scratch.loaders.dataLoader import DATALoaderV1


with open(r"data/raw/the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

tokenizer = tiktoken.get_encoding("gpt2")
enc_text = tokenizer.encode(raw_text)
print(len(enc_text))

print("-------------------")

enc_sample = enc_text[50:]

context_size = 4
x = enc_sample[:context_size]
y = enc_sample[1:context_size + 1]
print("x:", x)
print("y:", y)

print("-------------------")

for i in range(context_size):
    print(x[:i + 1], " ----> ", y[i])

print("-------------------")

for i in range(context_size):
    print(tokenizer.decode(x[:i + 1]), " ----> ", tokenizer.decode([y[i]]))
print("-------------------")


# Use of dataLoader
dataloader = DATALoaderV1().create_data_loader(
    txt=raw_text,
    batch_size=2,
    max_length=4,
    stride=1,
    shuffle=False,
)
data_iter = iter(dataloader)
first_batch = next(data_iter)
print(first_batch)
second_batch = next(data_iter)
print(second_batch)
print("-------------------")
