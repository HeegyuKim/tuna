from dataclasses import dataclass
from collections import defaultdict
from copy import deepcopy
from typing import Optional, Dict
from tqdm import tqdm
import random
from functools import partial

from task.base import DatasetArguments

from datasets import load_dataset, DatasetDict, concatenate_datasets, Value, Features, IterableDataset
from ..base import Dataset
from .processor import GPTDatasetArguments
from ..reward.dataset import PREDEFINED_FORMAT, join_conv, NUM_PROC
from .formatter import FORMATTERS


class LMDataset(Dataset):
    def __init__(self, args: GPTDatasetArguments, tokenizer, architecture: str = "causal-lm", for_generation=False, test_only=False) -> None:
        self.tokenizer = tokenizer
        self.tokenizer.truncation_side = 'right'
        self.prefix_tokenizer = deepcopy(tokenizer)
        self.prefix_tokenizer.truncation_side = 'left'
        self.architecture = architecture
        self.formatter = FORMATTERS[args.formatter]()
        self.for_generation = for_generation
        self.test_only = test_only
        super().__init__(args)

    def prepare_dataset(self, args):
        dataset = DatasetDict()
        if not self.test_only:
            dataset["train"] = self.load_multiple_dataset(args, args.dataset, "train")

        test_set = self.load_multiple_dataset(args, args.test_dataset or args.dataset, "test")
        if test_set:
            dataset["test"] = test_set

        if not self.for_generation and args.shuffle_data:
            dataset = dataset.shuffle(seed=42)

        if not args.dataset_streaming:
            if args.limit:
                for k, v in dataset.items():
                    dataset[k] = v.select(range(args.limit))
            
            dataset = self.encode_dataset(dataset)

            if not self.for_generation and args.packing:
                dataset = self.pack(dataset)
        else:
            for k, v in dataset.items():
                v = v.map(self.encode, load_from_cache_file=False)
                dataset[k] = IterableDataset.from_generator(
                    self.pack_iter,
                    gen_kwargs=dict(
                        iterable=v, 
                        limit=args.limit
                    )
                    )

        return dataset
    
    def encode_dataset(self, dataset):
        return dataset.map(self.encode, load_from_cache_file=False, num_proc=NUM_PROC)

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


    def load_multiple_dataset(self, args, datasets: str, split: str):
        datasets = datasets.split(",")
        num_datasets = len(datasets)
        final_datasets = []
        
        features = Features({'prompt': Value(dtype='string', id=None), 'context': [{'from': Value(dtype='string', id=None), 'value': Value(dtype='string', id=None)}], 'question': Value(dtype='string', id=None), 'chosen': Value(dtype='string', id=None), 'rejected': Value(dtype='string', id=None), 'source': Value(dtype='string', id=None)})
        columns = ["prompt", "context", "question", "chosen", "rejected", "source"]

        for ds in datasets:
            ds = self.load_dataset(args, ds, split)
            
            if ds is None:
                continue

            if args.limit:
                ds = ds.select(range(args.limit // num_datasets))

            ds = ds.select_columns(columns).cast(features)
            final_datasets.append(ds)

        if final_datasets:
            if num_datasets == 1:
                return final_datasets[0]
            else:
                return concatenate_datasets(final_datasets)
        else:
            return None
        
    def load_dataset(self, args, dataset: str, split: str):
        if ":" in dataset:
            format, name = dataset.split(":", 1)
            args = deepcopy(args)
            args.dataset = name
        else:
            format = dataset

        return PREDEFINED_FORMAT[format](args).load_and_parse(split)
    
    def encode_dataset(self, dataset):
        return dataset.map(self.encode, load_from_cache_file=False, num_proc=NUM_PROC)
    
    def build_text(self, prompt, context, question, answer):
        # return f"{prompt}\n\n{context}\n\nHuman: {question}\n\nAssistant:".strip()
        convs = [
            *context,
            {"from": "human", "value": question},
            {"from": "gpt", "value": answer},
        ]
        if prompt:
            convs.insert(0, {"from": "system", "value": prompt})
        elif self.args.default_prompt:
            convs.insert(0, {"from": "system", "value": self.args.default_prompt})

        text = self.formatter.format_uttrs(convs, self.tokenizer, self.for_generation)

        if self.args.generation_prefix:
            text[-1]["value"] = text[-1]["value"] + self.args.generation_prefix

        return text
    
    def encode(self, item):
        if self.architecture == "causal-lm":
            output = self.encode_causal_lm(item)
        elif self.architecture == "seq2seq":
            output = self.encode_seq2seq(item)
        else:
            raise
        output["source"] = item["source"]
        return output

    def encode_causal_lm(self, item):
        prompt, context, question, chosen = item["prompt"], item["context"], item["question"], item["chosen"]

        total_ids = []
        total_labels = []

        for conv in self.build_text(prompt, context, question, chosen):
            input_ids = self.tokenizer.encode(conv["value"], add_special_tokens=False)
            total_ids.extend(input_ids)

            if conv["from"] in ["human", "input", "system", "user"]:
                total_labels.extend([-100] * len(input_ids))
            else:
                total_labels.extend(input_ids)

        return {
            "input_ids": total_ids,
            "attention_mask": len(total_ids) * [1],
            "labels": total_labels
        }
    
    def encode_item(self, prefix, text, max_length: int, add_special_tokens: bool = False, truncate_prefix=True):
        tokenizer = self.prefix_tokenizer if truncate_prefix else self.tokenizer
        batch = tokenizer(text, truncation=True, max_length=max_length, add_special_tokens=add_special_tokens)
        return {f"{prefix}{k}": v for k, v in batch.items()}
    
    def encode_seq2seq(self, item):
        # prompt, context, question, chosen, rejected = item["tuple"]
        prompt, context, question, chosen, rejected = item["prompt"], item["context"], item["question"], item["chosen"], item["rejected"]

        prefix = self.build_text(prompt, context, question, "", True)
        prefix = self.formatter.join_convs(prefix, self.tokenizer)

        prefix = self.encode_item("", prefix, max_length=self.args.max_seq_length, add_special_tokens=False, truncate_prefix=True)
        chosen = self.encode_item("decoder_", chosen, max_length=self.args.max_decoder_seq_length, add_special_tokens=True, truncate_prefix=False)
        
        if chosen["decoder_input_ids"][-1] != self.tokenizer.eos_token_id:
            chosen["decoder_input_ids"][-1] = self.tokenizer.eos_token_id

        return dict(
            **prefix,
            **chosen
        )



class PPODataset(LMDataset):
    
    def encode(self, item):
        output = self.encode_causal_lm(item)
        output["source"] = item["source"]
        return output

    def encode_causal_lm(self, item):
        prompt, context, question, chosen = item["prompt"], item["context"], item["question"], item["chosen"]
        convs = self.build_text(prompt, context, question, chosen)
        model_prompt = "".join([x["value"] for x in convs])
        return {
            "model_prompt": model_prompt,
            "model_prompt_ids": self.tokenizer.encode(model_prompt)
        }
    