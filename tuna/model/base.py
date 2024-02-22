
from ..common import Registry
from dataclasses import dataclass
from typing import Optional
from transformers import AutoModel, AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer, AutoProcessor, LlavaForConditionalGeneration, LlavaProcessor, CLIPVisionModel, ClipImageProcessor
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

    def load_model_and_tokenizer(self):
        args = self.args
        tokenizer = args.tokenizer or args.model_name_or_path
        tokenizer = self.AUTO_TOKENIZER_CLASS.from_pretrained(tokenizer)
        model = self.AUTO_CLASS.from_pretrained(self.args.model_name_or_path)
        
        # LoRA
        if args.use_lora:
            model = self.apply_lora(args, model)

        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
            print("Setting pad token to eos token")
            
        artifacts = dict(
            tokenizer=tokenizer,
        )

        return model, artifacts
    
    def apply_lora(self, args, model, targets=None):
        targets = targets or find_lora_targets(model)

        from peft import LoraConfig, TaskType, get_peft_model
        from .lora_util import find_lora_targets

        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, 
            inference_mode=False, 
            r=args.lora_r, 
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



@dataclass
class LlavaForPretrainingArguments(BaseModelArguments):
    vision_tower: Optional[str] = None


@models("llava-for-pretrain")
class LlavaForPretrainingModel(BaseModel):
    ARG_CLASS = LlavaForPretrainingArguments
    AUTO_CLASS = LlavaForConditionalGeneration
    AUTO_TOKENIZER_CLASS = AutoProcessor

    def load_model_and_tokenizer(self):
        from transformers import LlavaConfig

        args = self.args
        tokenizer = args.tokenizer or args.model_name_or_path
        tokenizer = self.AUTO_TOKENIZER_CLASS.from_pretrained(tokenizer)
        lm = AutoModelForCausalLM.from_pretrained(self.args.model_name_or_path)

        if "<image>" not in tokenizer.special_tokens_map.values():
            print("inserting <image> token")
            smart_tokenizer_and_embedding_resize(
                {
                    "additional_special_tokens": ["<image>"]
                },
                tokenizer,
                lm
                )
        
        processor = AutoProcessor.from_pretrained(args.vision_tower)
        vision_tower = CLIPVisionModel.from_pretrained(args.vision_tower)

        image_token_index=tokenizer.convert_tokens_to_ids("<image>")
        assert isinstance(image_token_index, int)

        config = LlavaConfig(
            vision_config=vision_tower.config,
            text_config=lm.config,
            vocab_size=tokenizer.vocab_size,
            image_token_index=image_token_index,
        )

        freeze_model(vision_tower)
        # freeze_model(lm)

        model = LlavaForConditionalGeneration(config)
        model.vision_tower = vision_tower
        model.language_model = lm

        artifacts = dict(
            tokenizer=tokenizer,
            image_processor=processor.image_processor
        )
        return model, artifacts

@models("llava-for-finetune")
class LlavaForFinetuningModel(BaseModel):
    AUTO_CLASS = LlavaForConditionalGeneration
        
    def load_model_and_tokenizer(self):
        args = self.args
        tokenizer = args.tokenizer or args.model_name_or_path
        tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        model = LlavaForConditionalGeneration.from_pretrained(self.args.model_name_or_path)

        processor = ClipImageProcessor.from_pretrained(self.args.model_name_or_path)

        freeze_model(model.vision_tower)
        # freeze_model(lm)

        artifacts = dict(
            tokenizer=tokenizer,
            image_processor=processor
        )
        return model, artifacts
