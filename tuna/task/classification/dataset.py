
from task.base import DatasetArguments
from ..base import Dataset

from datasets import load_dataset


class YNAT(Dataset):
    
    def __init__(self, args: DatasetArguments, tokenizer) -> None:
        self.tokenizer = tokenizer
        super().__init__(args)

    def prepare_dataset(self, args):
        # with self.args.main_process_first():
        dataset = load_dataset("klue", "ynat")
        if args.limit:
            for k, v in dataset.items():
                dataset[k] = v.select(range(args.limit))
        # dataset = dataset.map(self.encode, remove_columns=dataset["train"].column_names, load_from_cache_file=True)
        dataset = dataset.map(self.encode, load_from_cache_file=True, num_proc=8)

        return {
            "train": dataset["train"],
            "test": dataset["validation"]
        }

    def encode(self, x):
        out = self.tokenizer(x['title'], truncation=True, max_length=self.args.max_seq_length)
        out["labels"] = x["label"]
        return out

class Emotion(Dataset):
    
    def __init__(self, args: DatasetArguments, tokenizer) -> None:
        self.tokenizer = tokenizer
        super().__init__(args)

    def prepare_dataset(self, args):
        # with self.args.main_process_first():
        dataset = load_dataset("dair-ai/emotion")
        for k, v in dataset.items():
            dataset[k] = v.select(range(32))
        # dataset = dataset.map(self.encode, remove_columns=dataset["train"].column_names, load_from_cache_file=True)
        dataset = dataset.map(self.encode, load_from_cache_file=True)

        return {
            "train": dataset["train"],
            "test": dataset["validation"]
        }

    def encode(self, x):
        out = self.tokenizer(x['text'], truncation=True, max_length=self.args.max_seq_length)
        out["labels"] = x["label"]
        return out