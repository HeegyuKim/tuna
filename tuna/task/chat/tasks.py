from dataclasses import dataclass
from datasets import DatasetDict
from ..base import LMTask, tasks, TaskArguments, TensorWrapper
from .train_templates import train_templates


@dataclass
class ChatLMTaskArguments(TaskArguments):
    train_template: str = "default"
    train_only_response: bool = True

@tasks.register("chat-lm")
class ChatLMTask(LMTask):
    ARG_CLASS = ChatLMTaskArguments

    def __init__(self, args, model, tokenizer, wrapper: TensorWrapper | str = ...) -> None:
        super().__init__(args, model, tokenizer, wrapper)
        self.train_template = train_templates[args.train_template](tokenizer)
    
    def encode_item(self, item):
        conversation = item["conversations"]
        concat_inputs, concat_labels = [], []

        for i, uttr in enumerate(conversation):
            content, trainable = self.train_template.handle_utterance(uttr, i)

            input_ids = self.tokenizer.encode(content, add_special_tokens=False)
            if self.args.train_only_response or trainable:
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
        

