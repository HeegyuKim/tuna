from vllm import LLM, SamplingParams
from tuna.serve.prompt_templates import PROMPT_TEMPLATES

from copy import deepcopy

class VLLMModel:
    def __init__(self, model_name: str, device: str = "cuda", dtype = "bfloat16", max_length: int = 4096):
        self.model = LLM(model=model_name, dtype=dtype, device=device, max_model_len=max_length)
        self.tokenizer = self.model.get_tokenizer()
        
        self.stop_tokens = list(set([self.tokenizer.eos_token, "<|endoftext|>", "[INST]", "[/INST]", "<|im_end|>", "<|end|>", "<|eot_id|>", "<end_of_turn>", "<start_of_turn>"]))

    def load_chat_template(self, chat_template: str):
        self.tokenizer.chat_template = PROMPT_TEMPLATES[chat_template]

    def gen_args_to_sampling_params(self, gen_args):
        return SamplingParams(
            temperature=gen_args.get("temperature", 1.0),
            top_k=gen_args.get("top_k", -1),
            top_p=gen_args.get("top_p", 1.0),
            max_tokens=gen_args.get("max_new_tokens"),
            eos_token=self.stop_tokens,
        )

    def generate_batch(self, prompts, histories=None, generation_prefix: str = None, gen_args={}):
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
                inputs = generation_prefix + inputs
            
            final_prompts.append(inputs)
        
        sampling_params = self.gen_args_to_sampling_params(gen_args)
        outputs = self.model.generate(final_prompts, sampling_params)
        
        return [output.outputs[0].text for output in outputs]
    
    def generate(self, prompt, history=None, generation_prefix: str = None, gen_args={}):
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
        
        sampling_params = self.gen_args_to_sampling_params(gen_args)
        outputs = self.model.generate([inputs], sampling_params)
        
        return outputs[0].outputs[0].text