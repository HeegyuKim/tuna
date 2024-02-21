
from typing import List, Dict, Any, Optional, Iterable, Union
from dataclasses import dataclass
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import Dataset, DatasetDict, concatenate_datasets


from ..common import Registry
from .collator import GenerativeLanguageModelCollator
from .dataset import DatasetArguments, DataSource, datasources, DatasetLoader


@dataclass
class TaskArguments:
    padding: Optional[str] = "longest"
    padding_side: Optional[str] = "right"

    truncation: bool = False
    truncation_side: Optional[str] = "right"
    packing: bool = False

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


class Task:
    ARG_CLASS = TaskArguments

    def __init__(self,
                 args,
                 model,
                 artifacts,
                 wrapper: Union[TensorWrapper, str] = TensorWrapper("cpu")
                 ) -> None:
        self.args = args
        self.model = model
        self.tokenizer = artifacts.get("tokenizer")
        
        if isinstance(wrapper, TensorWrapper):
            self.wrapper = wrapper
        else:
            self.wrapper = TensorWrapper(wrapper)

    def encode_datasets(self, datasets: DatasetDict) -> DatasetDict:
        datasets = datasets.map(self.encode_item, load_from_cache_file=False)
        return datasets

    def get_trainable_parameters(self):
        return self.model.parameters()
    
    def encode_item(self, item):
        pass

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

@tasks.register("lm")
class LMTask(Task):
    
    def __init__(self, args, model, artifacts, wrapper: Union[TensorWrapper, str] = TensorWrapper("cpu")) -> None:
        super().__init__(args, model, artifacts, wrapper)
        self._init_collator()

    def _init_collator(self):
        self.collator = GenerativeLanguageModelCollator(
            self.tokenizer, 
            padding=self.args.padding,
            padding_side=self.args.padding_side,
            max_length=self.args.max_length,
            decoder_max_length=self.args.decoder_max_length,
            return_tensors="pt")
        
    def encode_datasets(self, datasets: DatasetDict) -> DatasetDict:
        datasets = super().encode_datasets(datasets)
        if self.args.packing:
            cols = datasets["train"].column_names
            if "input_ids" in cols:
                cols.remove("input_ids")
            if "labels" in cols:
                cols.remove("labels")
            datasets = datasets.map(self._pack, load_from_cache_file=False, batched=True, remove_columns=cols, desc="Packing")
            
        return datasets

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
        if self.wrapper.is_xla:
            batch = self.wrapper(batch)
            
        if self.args.packing:
            return self.packed_step(batch, step)
        outputs = self.model(**batch)
        loss = outputs.loss
        print(batch)
        print(loss)
        return {"loss": loss}
    
    def packed_step(self, batch, step):
        labels = batch.pop("labels")
        outputs = self.model(**batch)
        logits = outputs.logits
        # cross entropy loss
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100)
        return {"loss": loss}

    def collate_step_outputs(self, outputs):
        loss = torch.stack([x["loss"] for x in outputs]).mean()
        return {"loss": loss}

    @property
    def eval_metric_definitions(self):
        return {"loss": "min"}
    
