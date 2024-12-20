
from typing import List, Dict, Any, Optional, Iterable, Union
from dataclasses import dataclass
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import Dataset, DatasetDict, concatenate_datasets, IterableDataset


from ..common import Registry
from .collator import GenerativeLanguageModelCollator
from .dataset import DatasetArguments, DataSource, datasources, DatasetLoader, NUM_PROC


@dataclass
class TaskArguments:
    padding: Optional[str] = "longest"
    padding_side: Optional[str] = "right"
    check_dataset: bool = True

    truncation: bool = False
    truncation_side: Optional[str] = "right"
    packing: bool = False
    packing_strategy: str = "reuse" # reuse, truncate, pad

    max_length: Optional[int] = None
    decoder_max_length: Optional[int] = None



def collate_dictlist(dl):
    out = defaultdict(list)

    for d in dl:
        for k, v in d.items():
            out[k].append(v)

    return out

class TensorWrapper():
    def __init__(self, device) -> None:
        self.device = device
        self.is_xla = "xla" in (device if isinstance(device, str) else device.type)

    def __call__(self, tensor):
        if torch.is_tensor(tensor):
            return tensor.to(self.device)
        if isinstance(tensor, dict):
            for k, v in tensor.items():
                if torch.is_tensor(v):
                    tensor[k] = v.to(self.device)
            return tensor
    

tasks = Registry("tasks")


class BaseTask:

    def __init__(self,
                 args,
                 artifacts,
                 ) -> None:
        self.args = args

    def encode_datasets(self, datasets: Dict, dataset_args: DatasetArguments) -> DatasetDict:
        for k in datasets:
            ds = datasets[k]
            if isinstance(ds, IterableDataset):
                datasets[k] = ds.map(self.encode_item).select_columns(["input_ids", "attention_mask", "labels"])
            else:
                datasets[k] = ds.map(self.encode_item, load_from_cache_file=dataset_args.load_from_cache_file, desc="Encoding", num_proc=NUM_PROC)

        return datasets

    def set_model(self, model):
        self.model = model
        
    def get_trainable_parameters(self):
        return self.model.parameters()
    
    def encode_item(self, item):
        pass

    def truncate_dict(self, d):
        if not self.args.truncation:
            return d
        
        nd = {}
        if self.args.decoder_max_length:
            if self.model.config.is_encoder_decoder and self.args.decoder_max_length:
                decoder_max_length = self.args.decoder_max_length
        else:
            decoder_max_length = self.args.max_length
        max_length = self.args.max_length

        for k in d:
            arr = d[k]
            
            if k == "labels" or "decoder" in k:
                length = decoder_max_length
            else:
                length = max_length
                
            if self.args.truncation_side == "right":
                arr = arr[:length]
            else:
                arr = arr[-length:]
            
            nd[k] = arr
        
        return nd
    
class Task(BaseTask):
    ARG_CLASS = TaskArguments

    def __init__(self,
                 args,
                 artifacts,
                 wrapper: Union[TensorWrapper, str] = TensorWrapper("cpu")
                 ) -> None:
        self.args = args
        self.tokenizer = artifacts.get("tokenizer")
        
        if isinstance(wrapper, TensorWrapper):
            self.wrapper = wrapper
        else:
            self.wrapper = TensorWrapper(wrapper)

    def get_trainable_parameters(self):
        return self.model.parameters()

    def wrap_batch(self, batch):
        return self.wrapper(batch)

    def step(self, batch, step):
        raise NotImplemented()
    
    def train_step(self, batch, step):
        return self.step(batch, step)

    def evaluation_step(self, batch, step):
        return self.step(batch, step)
    
    def collate_batch(self, batch):
        return collate_dictlist(batch)

    def collate_step_outputs(self, outputs):
        outputs = collate_dictlist(outputs)
        metrics = {k: torch.stack(v).type(torch.float32).mean() for k, v in outputs.items() if torch.is_tensor(v[0])}
        return metrics
    
    def collate_train_step_outputs(self, outputs):
        return self.collate_step_outputs(outputs)

    def collate_evaluation_outputs(self, outputs):
        return self.collate_step_outputs(outputs)

    @property
    def eval_metric_definitions(self):
        return {}
    
    def save_artifacts(self, model, path, **kwargs):
        model.save_pretrained(path, **kwargs)
        self.tokenizer.save_pretrained(path, **kwargs)


@tasks.register("lm")
class LMTask(Task):
    
    def __init__(self, args, artifacts, wrapper: Union[TensorWrapper, str] = TensorWrapper("cpu")) -> None:
        super().__init__(args, artifacts, wrapper)
        self._init_collator()

    def _init_collator(self):
        self.collator = GenerativeLanguageModelCollator(
            self.tokenizer, 
            padding=self.args.padding,
            padding_side=self.args.padding_side,
            max_length=self.args.max_length,
            decoder_max_length=self.args.decoder_max_length,
            return_tensors="pt")
        
    def encode_datasets(self, datasets: Dict, dataset_args: DatasetArguments) -> DatasetDict:
        datasets = super().encode_datasets(datasets, dataset_args)
        if self.args.packing:
            cols = datasets["train"].column_names
            if cols:
                if "input_ids" in cols:
                    cols.remove("input_ids")
                if "labels" in cols:
                    cols.remove("labels")
                for k in datasets:
                    datasets[k] = datasets[k].map(self._pack, load_from_cache_file=dataset_args.load_from_cache_file, batched=True, remove_columns=cols, desc="Packing", num_proc=NUM_PROC)
            else: # iterable dataset
                for k in datasets:
                    datasets[k] = datasets[k].map(self._pack, batched=True)
            
        if self.args.check_dataset:
            for k in datasets:
                datasets[k] = self.check_dataset(k, datasets[k], dataset_args)
            
        return datasets

    def check_dataset(self, split, dataset, dataset_args):
        filtered_dataset = dataset.filter(self.filter_item, num_proc=NUM_PROC, load_from_cache_file=dataset_args.load_from_cache_file, desc=f"Checking {split} set")
        if not isinstance(dataset, IterableDataset):
            original_size = len(dataset)
            filtered_len = len(filtered_dataset)
            if original_size != filtered_len:
                print(f"Filtered: {filtered_len - original_size} items from {split} set: {original_size} -> {filtered_len}")
        return filtered_dataset

    def filter_item(self, item):
        if "labels" in item:
            trainables = sum(x >= 0 for x in item["labels"])
        else:
            trainables = 0

        # for DPO, ORPO
        if "chosen" in item:
            trainables = sum(x >= 0 for x in item["chosen"]["labels"])
        if "rejected" in item:
            trainables += sum(x >= 0 for x in item["rejected"]["labels"])

        return trainables > 0

    def _pack(self, items):
        outputs = dict(
            input_ids=[],
            attention_mask=[],
            labels=[]
        )
        accum_len = 0

        batch_len = self.args.max_length
        all_input_ids = items["input_ids"]
        all_attention_mask = items.get("attention_mask")
        all_labels = items["labels"]

        batch_ids, batch_mask, batch_labels = [], [], []

        for ids, mask, labels in zip(all_input_ids, all_attention_mask, all_labels):
            accum_len += len(ids)

            batch_ids.extend(ids)
            if all_attention_mask is not None:
                batch_mask.extend(mask)
            batch_labels.extend(labels)

            while accum_len > batch_len:
                outputs["input_ids"].append(batch_ids[:batch_len])
                if all_attention_mask is not None:
                    outputs["attention_mask"].append(batch_mask[:batch_len])
                outputs["labels"].append(batch_labels[1:batch_len + 1])
                # outputs["labels"].append(batch_labels[:batch_len])

                batch_ids, batch_labels = batch_ids[batch_len:], batch_labels[batch_len:]
                if all_attention_mask is not None:
                    batch_mask = batch_mask[batch_len:]
                accum_len -= batch_len
        
        if all_attention_mask is None:
            outputs.pop("attention_mask")
        
        return outputs
    
    def collate_batch(self, batch):
        return self.collator(batch)
    
    def step(self, batch, step):
        if self.args.packing:
            return self.packed_step(batch, step)
        outputs = self.model(**batch)
        accuracy = self.compute_masked_accuracy(outputs.logits, batch["labels"])
        loss = outputs.loss
        return {"loss": loss, "accuracy": accuracy}

    def compute_masked_accuracy(self, logits, labels):
        mask = labels != -100
        return (logits[:, :-1].argmax(-1) == labels[:, 1:]).float().sum() / (mask.sum() + 1e-8)
    
    def packed_step(self, batch, step):
        labels = batch.pop("labels")
        outputs = self.model(**batch)
        accuracy = self.compute_masked_accuracy(outputs.logits, labels)
        logits = outputs.logits
        # cross entropy loss
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100)
        return {"loss": loss, "accuracy": accuracy}

    def collate_step_outputs(self, outputs):
        loss = torch.stack([x["loss"] for x in outputs]).mean()
        accuracy = torch.stack([x["accuracy"] for x in outputs]).mean()
        return {"loss": loss, "accuracy": accuracy}

    @property
    def eval_metric_definitions(self):
        return {"loss": "min", "accuracy": "max"}
    
