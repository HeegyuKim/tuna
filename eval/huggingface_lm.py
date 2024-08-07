import transformers
import torch
from tuna.serve.prompt_templates import PROMPT_TEMPLATES
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline, TextIteratorStreamer
from copy import deepcopy
from threading import Thread


def load_huggingface_model_tokenizer(model_name: str, dtype: torch.dtype = "auto", device = "auto", trust_remote_code=False, merge_peft=False):

    if "@" not in model_name:
        repo_id, revision = model_name, None
    else:
        repo_id, revision = model_name.split("@")

    if repo_id.startswith("peft:"):
        import peft

        repo_id = repo_id[len("peft:"):]

        config = peft.PeftConfig.from_pretrained(repo_id, revision=revision, trust_remote_code=trust_remote_code)
        base_model_name_or_path, base_revision = config.base_model_name_or_path, config.revision

        model = transformers.AutoModelForCausalLM.from_pretrained(base_model_name_or_path, revision=base_revision, torch_dtype=dtype, device_map=device, trust_remote_code=trust_remote_code)  
        tokenizer = transformers.AutoTokenizer.from_pretrained(base_model_name_or_path, revision=base_revision, trust_remote_code=trust_remote_code)

        model = peft.PeftModel.from_pretrained(model, repo_id, revision=revision)
        if merge_peft:
            model = model.merge_and_unload()
    else:
        model = transformers.AutoModelForCausalLM.from_pretrained(repo_id, revision=revision, torch_dtype=dtype, device_map=device, trust_remote_code=trust_remote_code)
        tokenizer = transformers.AutoTokenizer.from_pretrained(repo_id, revision=revision, trust_remote_code=trust_remote_code)
    
    return model, tokenizer


class HuggingfaceModel():
    SUPPORT_STREAMING = True

    def __init__(self, model_name: str, device: str = "cuda", dtype = torch.bfloat16):
        self.model, self.tokenizer = load_huggingface_model_tokenizer(model_name)

        self.model.eval()
        self.tokenizer.padding_side = 'left'
        self.device = torch.device(device)
            
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            print(f"Set pad token to eos token: {self.tokenizer.pad_token}")

    def load_chat_template(self, chat_template: str):
        self.tokenizer.chat_template = PROMPT_TEMPLATES[chat_template]

    def generate_batch(self, prompts, histories = None, generation_prefix: str = None, gen_args = {}):
        if histories is None:
            histories = [[] for _ in prompts]
        else:
            histories = deepcopy(histories)

        final_prompts = []
        for prompt, history in zip(prompts, histories):
            history.append({
                'role': 'user',
                'content': prompt,
            })

            inputs = self.tokenizer.apply_chat_template(history, add_special_tokens=True, tokenize=False, add_generation_prompt=True)
            if generation_prefix is not None:
                inputs = inputs + generation_prefix
            
            final_prompts.append(inputs)
        
        inputs = self.tokenizer(final_prompts, return_tensors="pt", padding=True).to(self.device)
        outputs = self.model.generate(**inputs, **gen_args)[:, inputs['input_ids'].shape[1]:]
        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
    
    def generate(self, prompt, history = None, generation_prefix: str = None, gen_args = {}):
        if history is None:
            history = []
        else:
            history = deepcopy(history)

        history.append({
            'role': 'user',
            'content': prompt,
        })

        inputs = self.tokenizer.apply_chat_template(history, add_special_tokens=True, tokenize=False, add_generation_prompt=True)
        if generation_prefix is not None:
            inputs = inputs + generation_prefix
        
        inputs = self.tokenizer(inputs, return_tensors="pt", padding=True).to(self.device)
        outputs = self.model.generate(**inputs, **gen_args)
        # print(self.tokenizer.decode(
        #     outputs[0, inputs['input_ids'].shape[1]:], 
        #     skip_special_tokens=False
        #     ))
        return self.tokenizer.decode(
            outputs[0, inputs['input_ids'].shape[1]:], 
            skip_special_tokens=True
            )

    def generate_stream(self, prompt, history = None, generation_prefix: str = None, gen_args = {}):
        if history is None:
            history = []
        else:
            history = deepcopy(history)

        history.append({
            'role': 'user',
            'content': prompt,
        })

        inputs = self.tokenizer.apply_chat_template(history, add_special_tokens=True, tokenize=False, add_generation_prompt=True)
        if generation_prefix is not None:
            inputs = inputs + generation_prefix
        
        print(inputs)
        inputs = self.tokenizer(inputs, return_tensors="pt", padding=True).to(self.device)
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True)
        generation_kwargs = dict(inputs, streamer=streamer, **gen_args)
        
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()

        yield from streamer