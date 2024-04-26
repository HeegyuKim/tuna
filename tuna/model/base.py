
from ..common import Registry
from dataclasses import dataclass
from typing import Optional
from transformers import AutoModel, AutoModelForSequenceClassification, AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer, AutoProcessor, LlavaForConditionalGeneration, LlavaProcessor, CLIPVisionModel, CLIPImageProcessor
from .utils import smart_tokenizer_and_embedding_resize, freeze_model, unfreeze_model

models = Registry("models")


@dataclass
class BaseModelArguments:
    # model
    model_name_or_path: str = ""
    revision: Optional[str] = None
    peft_model_id: Optional[str] = None
    peft_revision: Optional[str] = None
    num_labels: int = 1
    
    tokenizer: Optional[str] = None

    # LoRA
    use_lora: bool = False
    use_dora: bool = False
    lora_r: int = 8
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    lora_bias: Optional[str] = 'none'

class BaseModel:
    ARG_CLASS = BaseModelArguments
    AUTO_CLASS = AutoModel
    AUTO_TOKENIZER_CLASS = AutoTokenizer

    def __init__(self, args) -> None:
        self.args = args

    def load_artifacts(self):
        args = self.args
        tokenizer = args.tokenizer or args.model_name_or_path
        tokenizer = self.AUTO_TOKENIZER_CLASS.from_pretrained(tokenizer, revision=args.revision)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
            print("Setting tokenizer pad token to eos token")
            
        return dict(
            tokenizer=tokenizer,
            model_name_or_path=args.model_name_or_path,
        )

    def load_model(self):
        model = self.AUTO_CLASS.from_pretrained(self.args.model_name_or_path, revision=self.args.revision)
        
        # if model.config.pad_token_id is None:
        #     model.config.pad_token_id = model.config.eos_token_id
        #     print("Setting model pad token to eos token")

        # LoRA
        if self.args.use_lora or self.args.use_dora:
            model = self.apply_lora(self.args, model)
            
        return model
    
    def apply_lora(self, args, model, targets=None):
        from peft import LoraConfig, TaskType, get_peft_model
        from .lora_util import find_lora_targets
        targets = targets or find_lora_targets(model)


        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, 
            inference_mode=False, 
            r=args.lora_r, 
            use_dora=args.use_dora,
            lora_alpha=args.lora_alpha, 
            lora_dropout=args.lora_dropout, 
            target_modules=targets
            )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
        return model



@models("causal-lm")
class CausalLanguageModel(BaseModel):
    AUTO_CLASS = AutoModelForCausalLM

@models("seq2seq-lm")
class Seq2SeqLM(BaseModel):
    AUTO_CLASS = AutoModelForSeq2SeqLM

@models("sequence-classification")
class SequenceClassification(BaseModel):
    AUTO_CLASS = AutoModelForSequenceClassification

    def load_model(self):
        model = self.AUTO_CLASS.from_pretrained(self.args.model_name_or_path, num_labels=self.args.num_labels, revision=self.args.revision)
        
        # LoRA
        if self.args.use_lora or self.args.use_dora:
            model = self.apply_lora(self.args, model)
            
        return model

@dataclass
class LlavaForPretrainingArguments(BaseModelArguments):
    vision_tower: Optional[str] = None


@models("llava-for-pretrain")
class LlavaForPretrainingModel(BaseModel):
    ARG_CLASS = LlavaForPretrainingArguments
    AUTO_CLASS = LlavaForConditionalGeneration
    AUTO_TOKENIZER_CLASS = AutoProcessor

    def load_artifacts(self):
        args = self.args
        tokenizer = args.tokenizer or args.model_name_or_path
        tokenizer = self.AUTO_TOKENIZER_CLASS.from_pretrained(tokenizer, revision=args.revision)
        processor = AutoProcessor.from_pretrained(args.vision_tower, revision=args.revision)

        self.tokenizer = tokenizer

        return dict(
            model_name_or_path=args.model_name_or_path,
            tokenizer=tokenizer,
            image_processor=processor.image_processor
        )
        
    def load_model(self):
        from transformers import LlavaConfig

        args = self.args
        tokenizer = self.tokenizer
        lm = AutoModelForCausalLM.from_pretrained(self.args.model_name_or_path, revision=args.revision)

        if "<image>" not in tokenizer.special_tokens_map.values():
            print("inserting <image> token")
            smart_tokenizer_and_embedding_resize(
                {
                    "additional_special_tokens": ["<image>"]
                },
                tokenizer,
                lm
                )
        
        vision_tower = AutoModel.from_pretrained(args.vision_tower, revision=args.revision).vision_model

        image_token_index=tokenizer.convert_tokens_to_ids("<image>")
        assert isinstance(image_token_index, int)

        config = LlavaConfig(
            vision_config=vision_tower.config,
            text_config=lm.config,
            vocab_size=tokenizer.vocab_size,
            image_token_index=image_token_index,
        )

        freeze_model(vision_tower)
        freeze_model(lm)

        model = LlavaForConditionalGeneration(config)
        model.vision_tower = vision_tower
        model.language_model = lm

        return model
    

# @models("llava-for-finetune")
# class LlavaForFinetuningModel(BaseModel):
#     AUTO_CLASS = LlavaForConditionalGeneration

#     def load_artifacts(self):
#         args = self.args
#         tokenizer = args.tokenizer or args.model_name_or_path
#         tokenizer = AutoTokenizer.from_pretrained(tokenizer)
#         processor = LlavaProcessor.from_pretrained(self.args.model_name_or_path)

#         return dict(
#             tokenizer=tokenizer,
#             image_processor=processor
#         )
        
#     def load_model(self):
#         args = self.args
#         model = LlavaForConditionalGeneration.from_pretrained(self.args.model_name_or_path)

#         # freeze_model(model.vision_tower)
#         # freeze_model(lm)

#         return model
