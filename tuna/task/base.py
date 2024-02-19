
from typing import List, Dict, Any, Optional, Iterable, Union
from dataclasses import dataclass
from collections import defaultdict

import torch
import torch.nn as nn
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
                 tokenizer,
                 wrapper: Union[TensorWrapper, str] = TensorWrapper("cpu")
                 ) -> None:
        self.args = args
        self.model = model
        self.tokenizer = tokenizer
        
        if isinstance(wrapper, TensorWrapper):
            self.wrapper = wrapper
        else:
            self.wrapper = TensorWrapper(wrapper)

    def encode_datasets(self, datasets: DatasetDict) -> DatasetDict:
        datasets = datasets.map(self.encode_item, num_proc=8, load_from_cache_file=False)
        return datasets

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
    
    def __init__(self, args, model, tokenizer, wrapper: Union[TensorWrapper, str] = TensorWrapper("cpu")) -> None:
        super().__init__(args, model, tokenizer, wrapper)
        self.collator = GenerativeLanguageModelCollator(
            tokenizer, 
            padding=args.padding,
            padding_side=args.padding_side,
            max_length=args.max_length,
            decoder_max_length=args.decoder_max_length,
            return_tensors="pt")
    
    def collate_batch(self, batch):
        return self.collator(batch)
    
    def step(self, batch, step):
        outputs = self.model(**batch)
        loss = outputs.loss
        return {"loss": loss}

    def collate_step_outputs(self, outputs):
        loss = torch.stack([x["loss"] for x in outputs]).mean()
        return {"loss": loss}

    @property
    def eval_metric_definitions(self):
        return {"loss": "min"}
    
