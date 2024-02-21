from dataclasses import dataclass
from datasets import DatasetDict

import torch 

from ..base import LMTask, tasks, TaskArguments, TensorWrapper
from ..chat.tasks import ChatLMTask
from ..collator import GenerativeVLMCollator
from typing import Optional, Union


@tasks.register("chat-vlm")
class ChatVLMTask(ChatLMTask):

    def __init__(self, args, model, artifacts, wrapper: Union[TensorWrapper, str]) -> None:
        self.image_processor = artifacts["image_processor"]
        super().__init__(args, model, artifacts, wrapper)

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
        if "attention_mask" in batch:
            inputs_embeds, labels, attention_mask = self.build_input_embeds_for_xla(
                self.model, 
                None, 
                batch["pixel_values"], 
                batch["input_ids"], 
                batch["labels"], 
                batch["attention_mask"]
                )
        else:
            inputs_embeds, labels = self.build_input_embeds_for_xla(
                self.model, 
                None, 
                batch["pixel_values"], 
                batch["input_ids"], 
                batch["labels"], 
                )
            attention_mask = None

        outputs = self.model(
            inputs_embeds=inputs_embeds,
            labels=labels,
            attention_mask=attention_mask
            )
        return outputs
    
    def build_input_embeds_for_xla(self, model, prefix, pixel_values, texts, labels, attention_masks = None):
        embeddings = model.get_input_embeddings()

        if prefix:
            prefix_embeds = embeddings(prefix)

        texts_embeds = embeddings(texts)
        
        vision_feature_layer = model.config.vision_feature_layer
        vision_feature_select_strategy = model.config.vision_feature_select_strategy

        image_outputs = model.vision_tower(pixel_values, output_hidden_states=True)
        selected_image_feature = image_outputs.hidden_states[vision_feature_layer]
        # for i, hid in enumerate(image_outputs.hidden_states):
        #     print("image_outputs.hidden_states", i, hid.shape)

        selected_image_feature = model.multi_modal_projector(selected_image_feature)

        if vision_feature_select_strategy == "default":
            selected_image_feature = selected_image_feature[:, 1:]
        elif vision_feature_select_strategy == "full":
            selected_image_feature = selected_image_feature

        if prefix:
            input_embeds = torch.cat([prefix_embeds, selected_image_feature, texts_embeds], dim=1)
            labels = torch.cat([
                torch.full_like(prefix, -100),
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
            if prefix:
                attention_masks = torch.cat([
                    torch.ones_like(prefix),
                    torch.ones_like(selected_image_feature[:, :, 0]), attention_masks], 
                    dim=1
                    )
            else:
                attention_masks = torch.cat([
                    torch.ones_like(selected_image_feature[:, :, 0]), attention_masks], 
                    dim=1
                    )
                
            return input_embeds, labels, attention_masks
        else:
            return input_embeds, labels

    def save_artifacts(self, model, path, **kwargs):
        super().save_artifacts(model, path, **kwargs)
        self.image_processor.save_pretrained(path, **kwargs)


@tasks.register("llava-pretrain")
class LlavaPretrainingTask(ChatVLMTask):
    def get_trainable_parameters(self):
        return self.model.multi_modal_projector.parameters()

@tasks.register("llava-finetune")
class LlavaFinetuningTask(ChatVLMTask):
    def get_trainable_parameters(self):
        return list(self.model.multi_modal_projector.parameters()) + list(self.model.language_model.parameters())