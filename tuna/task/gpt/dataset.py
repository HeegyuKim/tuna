from dataclasses import dataclass
from collections import defaultdict
from copy import deepcopy
from typing import Optional, Dict
from tqdm import tqdm
import random
from functools import partial

from task.base import DatasetArguments

from datasets import load_dataset, DatasetDict, concatenate_datasets, Dataset as HDataset, IterableDataset, interleave_datasets
from ..base import Dataset
from .processor import GPTDatasetArguments, PROCESSOR_LIST


class GPTDataset(Dataset):
    def __init__(self, args: GPTDatasetArguments, tokenizer) -> None:
        self.tokenizer = tokenizer
        self.processor = PROCESSOR_LIST[args.processor](args)
        super().__init__(args)

    def prepare_dataset(self, args):
        dataset = self.processor.load_dataset(args)

        if args.shuffle_data:
            dataset = dataset.shuffle(seed=42)

        if not args.dataset_streaming:
            if args.limit:
                for k, v in dataset.items():
                    dataset[k] = v.select(range(args.limit))
            
            dataset = self.encode_dataset(dataset)

            if args.packing:
                dataset = self.pack(dataset)
        else:
            for k, v in dataset.items():
                v = v.map(self.encode)
                dataset[k] = IterableDataset.from_generator(
                    self.pack_iter,
                    gen_kwargs=dict(
                        iterable=v, 
                        limit=args.limit
                    )
                    )

        return dataset
    
    def encode_dataset(self, dataset):
        return dataset.map(self.encode, load_from_cache_file=False, num_proc=8)

    def encode(self, item):
        convs = self.processor.process(item, self.tokenizer)
        total_ids = []
        total_labels = []

        for conv in convs:
            input_ids = self.tokenizer.encode(conv["value"], add_special_tokens=False)
            total_ids.extend(input_ids)

            if conv["from"] in ["human", "input"]:
                total_labels.extend([-100] * len(input_ids))
            else:
                total_labels.extend(input_ids)

        return {
            "input_ids": total_ids,
            "attention_mask": len(total_ids) * [1],
            "labels": total_labels
        }


    def pack(self, dataset):
        packed_dataset = DatasetDict()
        for k, v in dataset.items():
            packed_dataset[k] = self._pack(v)
        return packed_dataset

    def _pack(self, dataset):
        return list(tqdm(self.pack_iter(dataset), "packing..."))

    def pack_iter(self, iterable, limit: Optional[int] = None):
        batch_ids, batch_labels = [], []
        max_len = self.args.max_seq_length
        count = 0

        for item in iterable:
            batch_ids.extend(item["input_ids"])
            batch_labels.extend(item["labels"])

            if len(batch_ids) > self.args.max_seq_length:
                yield {
                    "input_ids": batch_ids[:max_len],
                    "labels": batch_labels[1:max_len + 1],
                }
                batch_ids = batch_ids[max_len:]
                batch_labels = batch_labels[max_len:]

                count += 1

                if limit is not None and count >= limit:
                    return


