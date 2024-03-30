from dataclasses import dataclass
from datasets import DatasetDict
from ..base import LMTask, tasks, TaskArguments, TensorWrapper
from ..chat.train_templates import find_template
from ..collator import GenerativeVLMCollator
from typing import Optional, Union


@dataclass
class Seq2SeqLMTaskArguments(TaskArguments):
    train_template: Optional[str] = None
    train_prompt_prefix: Optional[str] = None

@tasks.register("seq2seq-lm")
class Seq2SeqLMTask(LMTask):
    ARG_CLASS = Seq2SeqLMTaskArguments

    def __init__(self, args, artifacts, wrapper: Union[TensorWrapper, str]) -> None:
        super().__init__(args, artifacts, wrapper)
        self.train_template = find_template(args.train_template or artifacts.get("model_name_or_path"))(self.tokenizer)
    
    def encode_item(self, item):
        input_text, output_text = item["input"], item["output"]
        
        if self.args.train_prompt_prefix:
            input_text = self.args.train_prompt_prefix + input_text

        input_ids = self.tokenizer.encode(input_text, add_special_tokens=False)
        labels = self.tokenizer.encode(output_text + self.tokenizer.eos_token, add_special_tokens=False)

        return self.truncate_dict({
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
            "labels": labels
        })
        
        