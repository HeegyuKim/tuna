from dataclasses import dataclass
from ..base import LMTask, tasks, TaskArguments, TensorWrapper
from ..chat.train_templates import find_template
from typing import Optional, Union


@dataclass
class PretrainLMTaskArguments(TaskArguments):
    train_template: Optional[str] = None

@tasks.register("pretrain-lm")
class PretrainLMTask(LMTask):
    ARG_CLASS = PretrainLMTaskArguments

    def __init__(self, args, model, artifacts, wrapper: Union[TensorWrapper, str]) -> None:
        super().__init__(args, model, artifacts, wrapper)
        self.train_template = find_template(args.train_template)(self.tokenizer)
    
    def encode_item(self, item):
        if "conversations" in item:
            return self.encode_conv_item(item)
        else:
            return self.encode_text_item(item)

    def encode_text_item(self, item):
        texts = item["text"]
        input_ids = self.tokenizer.encode(texts, add_special_tokens=False)
        labels = input_ids.copy()

        return {
            "input_ids": input_ids,
            "labels": labels
        }
    
    def encode_conv_item(self, item):
        conversation = item["conversations"]
        concat_inputs, concat_labels = [], []

        for i, uttr in enumerate(conversation):
            content, trainable = self.train_template.handle_utterance(uttr, i)

            input_ids = self.tokenizer.encode(content, add_special_tokens=False)

            concat_inputs.extend(input_ids)
            concat_labels.extend(input_ids)

        return self.truncate_dict({
            "input_ids": concat_inputs,
            # "attention_mask": [1] * len(concat_inputs),
            "labels": concat_labels
        })
        
        