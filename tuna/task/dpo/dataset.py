from dataclasses import dataclass
from collections import defaultdict
from copy import deepcopy
from typing import Optional, Dict
from tqdm import tqdm
import random
from functools import partial

from task.base import DatasetArguments

from ..reward.dataset import BaseRewardDataset, join_conv
    

class DPODataset(BaseRewardDataset):
    def build_text(self, prompt, context, question, answer):
        # return f"{prompt}\n\n{context}\n\nHuman: {question}\n\nAssistant:".strip()
        convs = [
            {"from": "system", "value": prompt},
            *context,
            {"from": "human", "value": question},
            {"from": "gpt", "value": answer},
        ]
        return join_conv(convs, self.args.join_method)
    
    def encode(self, item):
        if self.architecture == "causal-lm":
            output = self.encode_causal_lm(item)
        elif self.architecture == "seq2seq":
            output = self.encode_seq2seq(item)
        else:
            raise
        output["source"] = item.get("source")
        return output
    
    def encode_causal_lm(self, item):
        # prompt, context, question, chosen, rejected = item["tuple"]
        prompt, context, question, chosen, rejected = item["prompt"], item["context"], item["question"], item["chosen"], item["rejected"]

        context = self.build_text(prompt, context, question, "")

        chosen = self.encode_item(context, chosen + self.prefix_tokenizer.eos_token, max_length=self.args.max_seq_length)
        rejected = self.encode_item(context, rejected + self.prefix_tokenizer.eos_token, max_length=self.args.max_seq_length)
        
        return dict(
            chosen=chosen,
            rejected=rejected,
        )
    
    def encode_item(self, prompt, output, max_length: int, prefix="", add_special_tokens: bool = False, return_labels=True):
        batch_prompt = self.prefix_tokenizer(prompt, add_special_tokens=add_special_tokens)
        batch_output = self.tokenizer(output, add_special_tokens=add_special_tokens)

        batch = {}
        for k in batch_prompt.keys():
            batch[k] = batch_prompt[k] + batch_output[k]
            if len(batch[k]) > max_length:
                batch[k] = batch[k][-max_length:]

        input_ids = batch["input_ids"]
        
        if prefix:
            batch = {f"{prefix}{k}": v for k, v in batch.items()}

        if return_labels:
            output_length = len(batch_output["input_ids"])
            prefix_length = len(input_ids) - output_length
            
            batch = {k: v[1:] for k, v in batch.items()}
            batch["labels"] = [-100] * (prefix_length - 1) + input_ids[-output_length:]

        return batch