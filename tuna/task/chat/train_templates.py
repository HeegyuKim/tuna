from typing import Tuple, Dict, List
from ...common import Registry

train_templates = Registry("templates")

def find_template(model_id_or_template):
    if model_id_or_template in train_templates.keys():
        print(f"Select {model_id_or_template} template")
        return train_templates[model_id_or_template]

    for k in train_templates.keys():
        template = train_templates[k]
        if model_id_or_template in template.SUPPORTED_MODELS:
            print(f"Select {k} template")
            return template
        
    # default
    print("Select default template")
    return BaseTrainTemplate

@train_templates.register("default")
class BaseTrainTemplate:
    SUPPORTED_MODELS = []
    # for the first user message without system instruction (\eg Llama-2)
    INITIAL_USER_FORMAT = None

    SYSTEM_FORMAT = "<|im_start|>system\n{content}{eos}\n"
    USER_FORMAT = "<|im_start|>user\n{content}{eos}\n"
    ASSISTANT_FORMAT = "<|im_start|>assistant\n{content}{eos}\n"

    FUNCTION_CALLING_FORMAT = "<|im_start|>function_calling\n{content}{eos}\n"
    FUNCTION_RESPONSE_FORMAT = "<|im_start|>function_response\n{content}{eos}\n"


    def __init__(self, tokenizer) -> None:
        self.special_tokens = dict(
            eos=tokenizer.eos_token,
            bos=tokenizer.bos_token,
            pad=tokenizer.pad_token,
            sep=tokenizer.sep_token,
            cls=tokenizer.cls_token,
            mask=tokenizer.mask_token,
            unk=tokenizer.unk_token,
        )

    def handle_utterance(self, utterance: Dict, index: int) -> Tuple[str, bool]:
        role = utterance["role"]

        if role == "assistant":
            fmt = self.ASSISTANT_FORMAT
        elif role == "function-call":
            fmt = self.FUNCTION_CALLING_FORMAT
        elif role == "function-response":
            fmt = self.FUNCTION_RESPONSE_FORMAT
        elif role == "user":
            if index == 0 and self.INITIAL_USER_FORMAT:
                fmt = self.INITIAL_USER_FORMAT
            else:
                fmt = self.USER_FORMAT
        elif role == "system":
            fmt = self.SYSTEM_FORMAT
        else:
            raise ValueError(f"Unknown role: {role}")
        
        return fmt.format(content=utterance["content"], **self.special_tokens), role == "assistant"
        
    def join_utterances(self, utterances: List[str]) -> str:
        return "\n".join(utterances)

    def apply_chat_template(self, conversations):
        return self.join_utterances([self.handle_utterance(utt, i)[0] for i, utt in enumerate(conversations)])
        
        

@train_templates.register("tinyllama")
class TinyLlamaTemplate(BaseTrainTemplate):
    SUPPORTED_MODELS = [
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    ]
    # for the first user message without system instruction (\eg Llama-2)
    INITIAL_USER_FORMAT = None

    SYSTEM_FORMAT = "<|system|>\n{content}{eos}\n"
    USER_FORMAT = "<|user|>\n{content}{eos}\n"
    ASSISTANT_FORMAT = "<|assistant|>\n{content}{eos}\n"

    FUNCTION_CALLING_FORMAT = "<|function-call|>\n{content}{eos}\n"
    FUNCTION_RESPONSE_FORMAT = "<|function-response|>\n{content}{eos}\n"


@train_templates.register("42dot")
class HD42DotTemplate(BaseTrainTemplate):
    SUPPORTED_MODELS = [
        "42dot/42dot_LLM-SFT-1.3B"
    ]
    # for the first user message without system instruction (\eg Llama-2)
    INITIAL_USER_FORMAT = None

    SYSTEM_FORMAT = "{content}\n\n"
    USER_FORMAT = "<human>:\n{content}\n"
    ASSISTANT_FORMAT = "<bot>:\n{content}{eos}\n"

    FUNCTION_CALLING_FORMAT = "<function-call>:\n{content}{eos}\n"
    FUNCTION_RESPONSE_FORMAT = "<function-response>:\n{content}{eos}\n"



@train_templates.register("gemma")
class GemmaTemplate(BaseTrainTemplate):
    SUPPORTED_MODELS = [
        "google/gemma-2b-it",
        "google/gemma-7b-it"
    ]
    # for the first user message without system instruction (\eg Llama-2)
    INITIAL_USER_FORMAT = "<bos><start_of_turn>user\n{content}<end_of_turn>\n"

    SYSTEM_FORMAT = "<bos><start_of_turn>system{content}<end_of_turn>\n\n"
    USER_FORMAT = "<start_of_turn>user\n{content}<end_of_turn>\n"
    ASSISTANT_FORMAT = "<start_of_turn>model\n{content}<end_of_turn>\n"
    FUNCTION_CALLING_FORMAT = "<start_of_turn>function-call\n{content}<end_of_turn>\n"
    FUNCTION_RESPONSE_FORMAT = "<start_of_turn>function-response\n{content}<end_of_turn>\n"

@train_templates.register("gemma-vision")
class VisionGemmaTemplate(BaseTrainTemplate):
    # for the first user message without system instruction (\eg Llama-2)
    INITIAL_USER_FORMAT = "<start_of_turn>user\n{content}<end_of_turn>\n"

    SYSTEM_FORMAT = "<start_of_turn>system{content}<end_of_turn>\n\n"
    USER_FORMAT = "<start_of_turn>user\n{content}<end_of_turn>\n"
    ASSISTANT_FORMAT = "<start_of_turn>model\n{content}<end_of_turn>\n"
    FUNCTION_CALLING_FORMAT = "<start_of_turn>function-call\n{content}<end_of_turn>\n"
    FUNCTION_RESPONSE_FORMAT = "<start_of_turn>function-response\n{content}<end_of_turn>\n"