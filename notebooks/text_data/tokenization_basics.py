import re


# Loading data
with open(r"data/raw/the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()
print("total number of characters: ", len(raw_text))


# Split on white spaces.
text = "hello, world, this, is a test."
result = re.split(r"\s", text)
print(result)


# Split on punctuation also.
result = re.split(r"([.,]|\s)", text)
print(result)


# Remove redundant characters.
result = [item.strip() for item in result if item.strip()]
print(result)


# Support more punctuation characters.
text = "hello, world, -- this, is a test!."
result = re.split(r'([,.:;?_!"()\']|--|\s)', text)
result = [item.strip() for item in result if item.strip()]
print(result)


# Apply the same tokenization idea to the text file.
preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
preprocessed = [item.strip() for item in preprocessed if item.strip()]
print(len(preprocessed))
print(preprocessed[:30])

