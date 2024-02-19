from task.base import TensorWrapper
import torch.nn as nn
from ..base import Task, collate_dictlist
from trainer.utils import convert_dict_tensor_devices
from itertools import chain

import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F


def label_accuracy(logits, labels):
    # logits, labels = logits[:, :-1], labels[:, 1:]
    greedy = (logits.argmax(-1) == labels).float().sum(-1)
    label_tokens = (labels != -100).float().sum(-1)
    accuracy = torch.nan_to_num(greedy / label_tokens).mean()
    return accuracy


class GPTTask(Task):

    @property
    def eval_metric_definitions(self):
        return {
            "loss": "min",
        }

    def step(self, batch, step):
        inputs = self.wrapper(batch)
        inputs = {k: v for k, v in inputs.items() if torch.is_tensor(v)}

        labels = inputs.pop("labels")
        output = self.model(**inputs)
        
        batch_size, seq_len, vocab_size = output.logits.shape
        accuracy = label_accuracy(output.logits, labels)

        logits = output.logits.view(-1, vocab_size)
        labels = labels.view(-1)
        loss = F.cross_entropy(logits, labels, ignore_index=-100)

        return dict(
            loss=loss,
            accuracy=accuracy,
        )
