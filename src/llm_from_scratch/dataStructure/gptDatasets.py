import torch
import tiktoken
from torch.utils.data import Dataset


class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        super().__init__()
        self.input_ids = []
        self.target_ids = []

        tokenizer = tokenizer or tiktoken.get_encoding("gpt2")
        token_ids = tokenizer.encode(txt)

        # Each target sequence is shifted by one token for next-token prediction.
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1:i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, index):
        return self.input_ids[index], self.target_ids[index]
