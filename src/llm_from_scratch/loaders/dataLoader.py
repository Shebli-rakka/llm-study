import tiktoken
from torch.utils.data import DataLoader

from llm_from_scratch.dataStructure.gptDatasets import GPTDatasetV1


class DATALoaderV1:
    def create_data_loader(
        self,
        txt,
        batch_size=4,
        max_length=256,
        stride=128,
        shuffle=True,
        drop_last=True,
        num_workers=0,
    ):
        tokenizer = tiktoken.get_encoding("gpt2")
        dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=num_workers,
        )
