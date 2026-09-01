import re

from llm_study.tokenization.simple_tokenizer import (
    SimpleTokenizerV2,
    SimpleTokenizerV1,
)


# Loading data
with open(r"data/raw/the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()


# Tokenization
preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
preprocessed = [item.strip() for item in preprocessed if item.strip()]


# Building the vocabulary: individual token -> unique token id.
all_words = sorted(set(preprocessed))
vocab_size = len(all_words)
print(vocab_size)

vocab = {token: integer for integer, token in enumerate(all_words)}
for i, item in enumerate(vocab.items()):
    print(item)
    if i >= 50:
        break


# Using the vocab to encode text.
tokenizer = SimpleTokenizerV1(vocab)
text = """"It's the last he painted, you know,"
 Mrs. Gisburn said with pardonable pride."""
ids = tokenizer.encode(text)
print(ids)


# Decoding back to text.
print(tokenizer.decode(ids))


# Applying the tokenizer to new sample not contained in the vocab.

# text = "Hello, do you like tea?"
# print(tokenizer.encode(text)) 
# KeyError: 'Hello'


# Special context tokens: <unk> and <endoftext>.
all_tokens = sorted(set(preprocessed))
all_tokens.extend(["<|endoftext|>", "<|unk|>"])
vocab = {token: integer for integer, token in enumerate(all_tokens)}

print(len(vocab.items()))

for item in list(vocab.items())[-5:]:
    print(item)


# Tokenizer with special context.
text1 = "Hello, do you like tea?"
text2 = "In the sunlit terraces of the palace."

text = " <|endoftext|> ".join((text1, text2))
print(text)

tokenizer = SimpleTokenizerV2(vocab)
print(tokenizer.encode(text))
print(tokenizer.decode(tokenizer.encode(text)))
