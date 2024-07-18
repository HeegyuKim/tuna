from dataclasses import dataclass
from .flax_base import FlaxLMTask, flax_tasks, FlaxLMTaskArguments
from ..chat.train_templates import find_template
from typing import Optional, Union


@dataclass
class ChatLMTaskArguments(FlaxLMTaskArguments):
    train_template: Optional[str] = None
    train_only_response: bool = True
    train_prompt_prefix: Optional[str] = None

@flax_tasks.register("chat-lm")
class ChatLMTask(FlaxLMTask):
    ARG_CLASS = ChatLMTaskArguments


    def init_tokenizer_collator(self):
        super().init_tokenizer_collator()
        self.train_template = find_template(self.args.train_template or self.args.model_name_or_path)(self.tokenizer)
    
    def encode_item(self, item):
        conversation = item["conversations"]
        concat_inputs, concat_labels = [], []
        bos_token = self.tokenizer.bos_token
        
        if self.args.train_prompt_prefix:
            ids = self.tokenizer.encode(self.args.train_prompt_prefix, add_special_tokens=False)
            concat_inputs.extend(ids)
            concat_labels.extend([-100] * len(concat_inputs) if self.args.train_only_response else ids)

        for i, uttr in enumerate(conversation):
            content, trainable = self.train_template.handle_utterance(uttr, i)

            if self.args.packing and bos_token:
                content = content.replace(bos_token, "")

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
        
        