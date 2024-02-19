
from ..common import Registry
from dataclasses import dataclass
from typing import Optional
from transformers import AutoModel, AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer

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

    def __init__(self, args) -> None:
        self.args = args

    def load_model_and_tokenizer(self):
        args = self.args
        tokenizer = args.tokenizer or args.model_name_or_path
        tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        model = self.AUTO_CLASS.from_pretrained(self.args.model_name_or_path)
        
        # LoRA
        if args.use_lora:
            from peft import LoraConfig, TaskType, get_peft_model
            from .lora_util import find_lora_targets

            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM, 
                inference_mode=False, 
                r=args.lora_r, 
                lora_alpha=args.lora_alpha, 
                lora_dropout=args.lora_dropout, 
                target_modules=find_lora_targets(model)
                )
            model = get_peft_model(model, peft_config)
            model.print_trainable_parameters()

        return model, tokenizer
    

@models("causal-lm")
class CausalLanguageModel(BaseModel):
    AUTO_CLASS = AutoModelForCausalLM

@models("seq2seq-lm")
class Seq2SeqLM(BaseModel):
    AUTO_CLASS = AutoModelForSeq2SeqLM