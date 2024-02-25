from dataclasses import dataclass
from datasets import DatasetDict

import torch 

from ..base import LMTask, tasks, TaskArguments, TensorWrapper
from ..chat.tasks import ChatLMTask, ChatLMTaskArguments
from ..collator import GenerativeVLMCollator
from typing import Optional, Union


@dataclass
class ChatVLMArgs(ChatLMTaskArguments):
    image_prefix: Optional[str] = None

@tasks.register("chat-vlm")
class ChatVLMTask(ChatLMTask):
    ARG_CLASS = ChatVLMArgs

    def __init__(self, args, artifacts, wrapper: Union[TensorWrapper, str]) -> None:
        self.image_processor = artifacts["image_processor"]
        super().__init__(args, artifacts, wrapper)

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
        
        if self.args.image_prefix and not self.wrapper.is_xla:
            prompt_prefix = self.args.train_prompt_prefix or ""
            self.args.train_prompt_prefix = f"{prompt_prefix} {self.args.image_prefix}"

        # no multi-processing (OOM)
        datasets = datasets.map(self.encode_item, load_from_cache_file=False)
        return datasets
    
    
    def step(self, batch, step):
        if self.wrapper.is_xla:
            outputs = self.xla_step(batch, step)
        else:
            outputs = self.model(**batch)
        loss = outputs.loss
        return {"loss": loss}

    def xla_step(self, batch, step):
        batch = self.wrapper(batch)
        inputs_embeds, labels, attention_mask = self.build_input_embeds_for_xla(
            self.model, 
            batch["pixel_values"], 
            batch["input_ids"], 
            batch["labels"], 
            batch.get("attention_mask")
            )

        outputs = self.model(
            inputs_embeds=inputs_embeds,
            labels=labels,
            attention_mask=attention_mask
            )
        return outputs
    
    def build_input_embeds_for_xla(self, model, pixel_values, texts, labels, attention_masks = None):
        embeddings = model.get_input_embeddings()

        if self.args.image_prefix:
            image_prefix = [self.args.image_prefix] * texts.shape[0]
            image_prefix = self.wrapper(self.tokenizer(image_prefix, add_special_tokens=False, return_tensors="pt")["input_ids"])
            prefix_embeds = embeddings(image_prefix)
        else:
            prefix_embeds = None

        texts_embeds = embeddings(texts)
        
        vision_feature_layer = model.config.vision_feature_layer
        vision_feature_select_strategy = model.config.vision_feature_select_strategy

        # print("pixel_values", pixel_values.shape)
        # print("model vision vision", model.vision_tower.config)

        image_outputs = model.vision_tower(pixel_values, output_hidden_states=True)
        selected_image_feature = image_outputs.hidden_states[vision_feature_layer]

        selected_image_feature = model.multi_modal_projector(selected_image_feature)

        if vision_feature_select_strategy == "default":
            selected_image_feature = selected_image_feature[:, 1:]
        elif vision_feature_select_strategy == "full":
            selected_image_feature = selected_image_feature

        if prefix_embeds is not None:
            input_embeds = torch.cat([prefix_embeds, selected_image_feature, texts_embeds], dim=1)
            labels = torch.cat([
                torch.full(prefix_embeds.shape[:2], -100, device=texts.device), 
                torch.full(selected_image_feature.shape[:2], -100, device=texts.device), 
                texts
                ], dim=1)
        else:
            input_embeds = torch.cat([selected_image_feature, texts_embeds], dim=1)
            labels = torch.cat([
                torch.full(selected_image_feature.shape[:2], -100, device=texts.device), 
                texts
                ], dim=1)

        if attention_masks is not None:
            if prefix_embeds is not None:
                attention_masks = torch.cat([
                        torch.ones(prefix_embeds.shape[:2], device=texts.device),
                        torch.ones(selected_image_feature.shape[:2], device=texts.device),
                        attention_masks
                    ], 
                    dim=1
                    )
            else:
                attention_masks = torch.cat([
                        torch.ones(selected_image_feature.shape[:2], device=texts.device),
                        attention_masks
                    ], 
                    dim=1
                    )
                
        return input_embeds, labels, attention_masks

    def save_artifacts(self, model, path, **kwargs):
        super().save_artifacts(model, path, **kwargs)
        self.image_processor.save_pretrained(path, **kwargs)


@tasks.register("llava-for-pretrain")
class LlavaPretrainingTask(ChatVLMTask):
    def get_trainable_parameters(self):
        return self.model.multi_modal_projector.parameters()

@tasks.register("llava-for-finetune")
class LlavaFinetuningTask(ChatVLMTask):
    def get_trainable_parameters(self):
        return list(self.model.multi_modal_projector.parameters()) + list(self.model.language_model.parameters())