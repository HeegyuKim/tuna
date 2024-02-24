from dataclasses import dataclass
from datasets import DatasetDict
from ..base import LMTask, tasks, TaskArguments, TensorWrapper
from .train_templates import find_template
from ..collator import GenerativeVLMCollator
from typing import Optional, Union


@dataclass
class ChatLMTaskArguments(TaskArguments):
    train_template: Optional[str] = None
    train_only_response: bool = True
    train_prompt_prefix: Optional[str] = None

@tasks.register("chat-lm")
class ChatLMTask(LMTask):
    ARG_CLASS = ChatLMTaskArguments

    def __init__(self, args, artifacts, wrapper: Union[TensorWrapper, str]) -> None:
        super().__init__(args, artifacts, wrapper)
        self.train_template = find_template(args.train_template or artifacts.get("model_name_or_path"))(self.tokenizer)
    
    def encode_item(self, item):
        conversation = item["conversations"]
        concat_inputs, concat_labels = [], []
        
        if self.args.train_prompt_prefix:
            ids = self.tokenizer.encode(self.args.train_prompt_prefix, add_special_tokens=False)
            concat_inputs.extend(ids)
            concat_labels.extend([-100] * len(concat_inputs) if self.args.train_only_response else ids)

        for i, uttr in enumerate(conversation):
            content, trainable = self.train_template.handle_utterance(uttr, i)

            input_ids = self.tokenizer.encode(content, add_special_tokens=False)
            if not self.args.train_only_response or trainable:
                labels = input_ids.copy()
            else:
                labels = [-100] * len(input_ids)

            concat_inputs.extend(input_ids)
            concat_labels.extend(labels)

        return self.truncate_dict({
            "input_ids": concat_inputs,
            "attention_mask": [1] * len(concat_inputs),
            "labels": concat_labels
        })
        
        