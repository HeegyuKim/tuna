from dataclasses import dataclass
from datasets import DatasetDict
from ..base import LMTask, tasks, TaskArguments, TensorWrapper
from ..collator import GenerativeVLMCollator
from typing import Optional, Union, List, Dict, Any
from transformers import DataCollatorWithPadding, PreTrainedTokenizerBase
from transformers.tokenization_utils import PaddingStrategy
import torch



@dataclass
class SequenceClassificationDataCollator:
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"
    allowed_keys = [
        "input_ids",
        "attention_mask",
        "token_type_ids",
        "labels",
        "label_ids",
    ]

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        features = [{k: v for k, v in x.items() if k in self.allowed_keys} for x in features]
        batch = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        batch["attention_mask"] = batch["input_ids"].ne(self.tokenizer.pad_token_id).long()
        if "label" in batch:
            batch["labels"] = batch["label"]
            del batch["label"]
        if "label_ids" in batch:
            batch["labels"] = batch["label_ids"]
            del batch["label_ids"]
        return batch



@dataclass
class SeqClassificationTaskArguments(TaskArguments):
    pass

@tasks.register("sequence-classification")
class SeqClassificationTask(LMTask):
    ARG_CLASS = SeqClassificationTaskArguments

    def __init__(self, args, artifacts, wrapper: TensorWrapper | str = ...) -> None:
        super().__init__(args, artifacts, wrapper)
        self.tokenizer.truncation_side = args.truncation_side

    def _init_collator(self):
        self.collator = SequenceClassificationDataCollator(
            self.tokenizer, 
            padding=self.args.padding,
            # padding_side=self.args.padding_side,
            max_length=self.args.max_length,
            # decoder_max_length=self.args.decoder_max_length,
            return_tensors="pt")
        

    def encode_item(self, item):
        text = item["text"]
        input_ids = self.tokenizer.encode(
            text, 
            add_special_tokens=True, 
            truncation=self.args.truncation, 
            max_length=self.args.max_length, 
            padding=self.args.padding
            )

        return {
            "input_ids": input_ids,
            # "attention_mask": [1] * len(input_ids),
            "labels": item["label"]
        }

    def step(self, batch, step):
        if self.wrapper.is_xla:
            batch = self.wrapper(batch)
            
        outputs = self.model(**batch)
        acc = outputs.logits.argmax(dim=-1).eq(batch["labels"]).float().mean()
        loss = outputs.loss

        if step < 5:
            print(self.tokenizer.batch_decode(batch["input_ids"]))
            print(batch)
            print(outputs.logits.shape)

        return {
            "loss": loss,
            "accuracy": acc
            }
    

    def collate_step_outputs(self, outputs):
        loss = torch.stack([x["loss"] for x in outputs]).mean()
        acc = torch.stack([x["accuracy"] for x in outputs]).mean()
        return {
            "loss": loss,
            "accuracy": acc
            }

    @property
    def eval_metric_definitions(self):
        return {"loss": "min", "accuracy": "max"}