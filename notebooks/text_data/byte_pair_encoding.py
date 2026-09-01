import tiktoken

# from importlib.metadata import version
# print(version("tiktoken"))

tokenizer = tiktoken.get_encoding("gpt2")

text = (
    "Hello, do you like tea? <|endoftext|> In the sunlit terraces "
    "of someunknownPlace."
)
integers = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
print(integers)

strings = tokenizer.decode(integers)
print(strings)

word = "Akwirw ier"
word_tokens = tokenizer.encode(word, allowed_special="all")
print(word_tokens)

for token in word_tokens:
    print(tokenizer.decode([token]))

print(tokenizer.decode(word_tokens))
