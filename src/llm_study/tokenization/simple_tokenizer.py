import re


def _split_text(text, punctuation):
    preprocessed = re.split(punctuation, text)
    return [item.strip() for item in preprocessed if item.strip()]


def _tokens_to_text(ids, int_to_str):
    text = " ".join([int_to_str[i] for i in ids])
    return re.sub(r'\s+([,.?!"()\'])', r'\1', text)


class SimpleTokenizerV1:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i: s for s, i in vocab.items()}

    def encode(self, text):
        preprocessed = _split_text(text, r'([,.?_!"()\']|--|\s)')
        return [self.str_to_int[token] for token in preprocessed]

    def decode(self, ids):
        return _tokens_to_text(ids, self.int_to_str)


class SimpleTokenizerV2:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i: s for s, i in vocab.items()}

    def encode(self, text):
        preprocessed = _split_text(text, r'([,.:;?_!"()\']|--|\s)')
        preprocessed = [
            item if item in self.str_to_int else "<|unk|>"
            for item in preprocessed
        ]
        return [self.str_to_int[s] for s in preprocessed]

    def decode(self, ids):
        return _tokens_to_text(ids, self.int_to_str)
