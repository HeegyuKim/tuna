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

    SYSTEM_FORMAT = "<|im_start|>system\n{content}{eos}"
    USER_FORMAT = "<|im_start|>user\n{content}{eos}"
    ASSISTANT_FORMAT = "<|im_start|>assistant\n{content}{eos}"
    GENERATION_PROMPT = "<|im_start|>assistant"

    FUNCTION_CALLING_FORMAT = "<|im_start|>function_calling\n{content}{eos}"
    FUNCTION_RESPONSE_FORMAT = "<|im_start|>function_response\n{content}{eos}"

    TURN_SEPERATOR = "\n"

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
        
        if "trainable" in utterance and utterance["trainable"] is not None:
            trainable = utterance["trainable"]
        else:
            trainable = role == "assistant"

        return fmt.format(content=utterance["content"], **self.special_tokens), trainable
        
    def join_utterances(self, utterances: List[str]) -> str:
        return self.TURN_SEPERATOR.join(utterances)

    def apply_chat_template(self, conversations, add_generation_prompt=False):
        output = self.join_utterances([self.handle_utterance(utt, i)[0] for i, utt in enumerate(conversations)])
        if add_generation_prompt:
            output = output + self.GENERATION_PROMPT
        return output
        
@train_templates.register("chatml")
class ChatMLTrainTemplate(BaseTrainTemplate):
    pass

@train_templates.register("default:bos")
class BaseBOSTrainTemplate(BaseTrainTemplate):
    INITIAL_USER_FORMAT = "{bos}<|im_start|>user\n{content}{eos}"

@train_templates.register("tinyllama")
class TinyLlamaTemplate(BaseTrainTemplate):
    SUPPORTED_MODELS = [
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    ]
    # for the first user message without system instruction (\eg Llama-2)
    INITIAL_USER_FORMAT = None

    SYSTEM_FORMAT = "<|system|>\n{content}{eos}"
    USER_FORMAT = "<|user|>\n{content}{eos}"
    ASSISTANT_FORMAT = "<|assistant|>\n{content}{eos}"
    GENERATION_PROMPT = "<|assistant|>"

    FUNCTION_CALLING_FORMAT = "<|function-call|>\n{content}{eos}"
    FUNCTION_RESPONSE_FORMAT = "<|function-response|>\n{content}{eos}"


@train_templates.register("llama")
class LlamaTemplate(BaseTrainTemplate):
    SUPPORTED_MODELS = [
        "meta-llama/Llama-2-7b-chat-hf",
        "openbmb/Eurus-7b-sft",
        "openbmb/Eurus-7b-kto"
    ]
    # for the first user message without system instruction (\eg Llama-2)
    INITIAL_USER_FORMAT = "{bos}[INST] {content} [/INST]"
    SYSTEM_FORMAT = "{bos}[INST] <<SYS>>\n{content}<</SYS>>\n"
    SECOND_USER_FORMAT = "\n{content} [/INST]"
    USER_FORMAT = "[INST] {content} [/INST]"
    ASSISTANT_FORMAT = "{content}{eos}"
    GENERATION_PROMPT = ""

    FUNCTION_CALLING_FORMAT = "[function-call] {content}{eos}"
    FUNCTION_RESPONSE_FORMAT = "[function-response] {content}{eos}"

    TURN_SEPERATOR = " "

    def handle_utterance(self, utterance: Dict, index: int) -> Tuple[str, bool]:
        role = utterance["role"]

        if role == "assistant":
            fmt = self.ASSISTANT_FORMAT
        elif role == "function-call":
            fmt = self.FUNCTION_CALLING_FORMAT
        elif role == "function-response":
            fmt = self.FUNCTION_RESPONSE_FORMAT
        elif role == "user":
            if index == 0:
                fmt = self.INITIAL_USER_FORMAT
            if index == 1:
                fmt = self.SECOND_USER_FORMAT
            else:
                fmt = self.USER_FORMAT
        elif role == "system":
            fmt = self.SYSTEM_FORMAT
        else:
            raise ValueError(f"Unknown role: {role}")
        
        if "trainable" in utterance and utterance["trainable"] is not None:
            trainable = utterance["trainable"]
        else:
            trainable = role == "assistant"

        return fmt.format(content=utterance["content"], **self.special_tokens), trainable
        

@train_templates.register("zephyr")
class ZephyrTemplate(TinyLlamaTemplate):
    SUPPORTED_MODELS = [
        "HuggingFaceH4/mistral-7b-sft-beta",
        "HuggingFaceH4/zephyr-7b-beta"
    ]

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
    GENERATION_PROMPT = "<bot>:"

    FUNCTION_CALLING_FORMAT = "<function-call>:\n{content}{eos}\n"
    FUNCTION_RESPONSE_FORMAT = "<function-response>:\n{content}{eos}\n"


@train_templates.register("llama3")
class Llama3(BaseTrainTemplate):
    SUPPORTED_MODELS = [
        "beomi/Llama-3-Open-Ko-8B"
    ]

    INITIAL_USER_FORMAT = "{bos}<start_of_turn>user\n{content}<end_of_turn>"
    SYSTEM_FORMAT = "{bos}<|start_header_id|>system<|end_header_id|>\n\n{content}<|eot_id|>"
    USER_FORMAT = "<|start_header_id|>user<|end_header_id|>\n\n{content}<|eot_id|>\n"
    ASSISTANT_FORMAT = "<|start_header_id|>assistant<|end_header_id|>\n\n{content}<|eot_id|>"
    GENERATION_PROMPT = "<|start_header_id|>assistant<|end_header_id|>\n\n"

    FUNCTION_CALLING_FORMAT = "<|start_header_id|>function-call<|end_header_id|>\n\n{content}<|eot_id|>"
    FUNCTION_RESPONSE_FORMAT = "<|start_header_id|>function-response<|end_header_id|>\n\n{content}<|eot_id|>"

    def __init__(self, tokenizer) -> None:
        super().__init__(tokenizer)
        additional_tokens = ['<|start_header_id|>', '<|end_header_id|>', '<|eot_id|>']

        tokenizer.add_special_tokens({
            'additional_special_tokens': additional_tokens
            })

        tokenizer.eos_token = "<|eot_id|>"


@train_templates.register("gemma")
class GemmaTemplate(BaseTrainTemplate):
    SUPPORTED_MODELS = [
        "google/gemma-2b-it",
        "google/gemma-7b-it"
    ]
    # for the first user message without system instruction (\eg Llama-2)
    INITIAL_USER_FORMAT = "<bos><start_of_turn>user\n{content}<end_of_turn>"

    SYSTEM_FORMAT = "<bos><start_of_turn>system{content}<end_of_turn>"
    USER_FORMAT = "<start_of_turn>user\n{content}<end_of_turn>"
    ASSISTANT_FORMAT = "<start_of_turn>model\n{content}<end_of_turn>"
    GENERATION_PROMPT = "<start_of_turn>model"
    FUNCTION_CALLING_FORMAT = "<start_of_turn>model\n```function-call\n{content}```<end_of_turn>"
    FUNCTION_RESPONSE_FORMAT = "<start_of_turn>user\n```function-response\n{content}```<end_of_turn>"
    
    def __init__(self, tokenizer) -> None:
        super().__init__(tokenizer)
        additional_tokens = ['<start_of_turn>','<end_of_turn>']
        tokenizer.add_special_tokens({
            'additional_special_tokens': additional_tokens
            })
        
@train_templates.register("gemma-vision")
class VisionGemmaTemplate(BaseTrainTemplate):
    # for the first user message without system instruction (\eg Llama-2)
    INITIAL_USER_FORMAT = "<start_of_turn>user\n{content}<eos>\n"

    SYSTEM_FORMAT = "<start_of_turn>system{content}<eos>\n\n"
    USER_FORMAT = "<start_of_turn>user\n{content}<eos>\n"
    ASSISTANT_FORMAT = "<start_of_turn>model\n{content}<eos>\n"
    GENERATION_PROMPT = "<start_of_turn>model"
    
    FUNCTION_CALLING_FORMAT = "<start_of_turn>function-call\n{content}<eos>\n"
    FUNCTION_RESPONSE_FORMAT = "<start_of_turn>function-response\n{content}<eos>\n"


if __name__ == "__main__":
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("beomi/Llama-3-Open-Ko-8B")
    # print(tokenizer.encode("<end_of_turn>"))
    t = Llama3(tokenizer)
    # print(tokenizer.encode("<end_of_turn>"))
    convs = [
        {
            "role": "user",
            "content": "Hi"
        },
        {
            "role": "assistant",
            "content": "Hi"
        },
        {
            "role": "user",
            "content": "Hi22"
        },
        {
            "role": "assistant",
            "content": "Hi22"
        },
    ]
    out = t.apply_chat_template(convs)

    print(out)

    for i, c in enumerate(convs):
        print(f"uttr #{i}")
        uttr, _ = t.handle_utterance(c, i)
        print(uttr)
        ids = tokenizer.encode(uttr, add_special_tokens=False)
        print(ids)
        print(tokenizer.decode(ids, skip_special_tokens=False))