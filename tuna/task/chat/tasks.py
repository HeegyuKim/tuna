from dataclasses import dataclass
from datasets import DatasetDict
from ..base import LMTask, tasks, TaskArguments, TensorWrapper
from .train_templates import find_template
from ..collator import GenerativeVLMCollator
from typing import Optional


@dataclass
class ChatLMTaskArguments(TaskArguments):
    train_template: Optional[str] = None
    train_only_response: bool = True

@tasks.register("chat-lm")
class ChatLMTask(LMTask):
    ARG_CLASS = ChatLMTaskArguments

    def __init__(self, args, model, tokenizer, wrapper: TensorWrapper | str = ...) -> None:
        super().__init__(args, model, tokenizer, wrapper)
        self.train_template = find_template(args.train_template)(tokenizer)
    
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
        

@tasks.register("chat-vlm")
class ChatVLMTask(ChatLMTask):
    ARG_CLASS = ChatLMTaskArguments

    def __init__(self, args, model, processor, wrapper: TensorWrapper | str = ...) -> None:
        self.image_processor = processor.image_processor
        super().__init__(args, model, processor.tokenizer, wrapper)

    def _init_collator(self):
        self.collator = GenerativeVLMCollator(
            self.tokenizer, 
            padding=self.args.padding,
            padding_side=self.args.padding_side,
            max_length=self.args.max_length,
            decoder_max_length=self.args.decoder_max_length,
            image_processor=self.image_processor,
            return_tensors="pt"
            )
        
    def encode_datasets(self, datasets: DatasetDict) -> DatasetDict:
        datasets = datasets.map(self.encode_item, load_from_cache_file=False)
        return datasets
    
    # def encode_item(self, item):
    #     pixel_values = self.image_processor(item["image"], return_tensors="pt")
    #     item = super().encode_item(item)
    #     item["pixel_values"] = pixel_values

    #     return item


@tasks.register("llava-pretrain")
class LlavaPretrainingTask(ChatVLMTask):
    def get_trainable_parameters(self):
        return self.model.multi_modal_projector.parameters()
        