import torch.nn as nn
from ..base import Task

import torch
import torch.nn.functional as F


class ClassificationTask(Task):

    def __init__(self, model: nn.Module, binary_threshold: float = 0.5) -> None:
        super().__init__(model)
        self.binary_threshold = binary_threshold

    def compute_metric(self, logits, labels):
        num_labels = logits.shape[-1]
        if num_labels > 1:
            loss = F.cross_entropy(logits, labels)
            accuracy = logits.argmax(-1) == labels
        else:
            loss = F.binary_cross_entropy(logits, labels)
            accuracy = (logits > self.binary_threshold).type(torch.int32) == labels
        
        accuracy = accuracy.float().mean()

        return dict(
            loss=loss,
            accuracy=accuracy
        )
        
    def step(self, batch, step):
        kwargs = dict(
            input_ids=batch["input_ids"],
            attention_mask=batch.get("attention_mask"),
        )
        if "token_type_ids" in batch:
            kwargs["token_type_ids"] = batch["token_type_ids"]

        logits = self.model(**kwargs).logits
        labels = batch["labels"]

        return self.compute_metric(logits, labels)