import os
import jsonlines


def estimate_skip_length(output_path: str):
    if os.path.exists(output_path):
        with jsonlines.open(output_path, "r") as f:
            skip_length = len(list(f))
    else:
        skip_length = 0

    return skip_length


def batched_iteration(iterable, batch_size):
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def load_model(
        model_name: str, 
        prompt_length: int = 1024, 
        max_length: int = 2048,
        eos_token: str = None,
        batch_size: int = 1,
        gen_args: dict = None,
        chat_template: str = None,
        use_vllm: bool = False,
        compile: bool = True
        ):
    if use_vllm:
        try:
            import vllm
            use_vllm = True
        except ImportError:
            use_vllm = False

    if use_vllm:
        from .vllm_wrapper import VLLMModel
        print("Using VLLM")
        model = VLLMModel(model_name)
        if chat_template:
            model.load_chat_template(chat_template)
        return model
    
    import torch
    if torch.cuda.is_available():
        print("Using CUDA")
        from .huggingface_lm import HuggingfaceModel
        model = HuggingfaceModel(model_name)
        if chat_template:
            model.load_chat_template(chat_template)
        if eos_token:
            model.tokenizer.eos_token = eos_token
        return model
    else:
        print("Using Flax")
        from tuna.serve.flax_generator import FlaxHuggingfaceModel
        model = FlaxHuggingfaceModel(
            model_name,
            prompt_length=prompt_length,
            max_length=max_length,
            fully_sharded_data_parallel=False,
            eos_token=eos_token,
            batch_size=batch_size,
            gen_args=gen_args,
            chat_template=chat_template,
            )
        if compile:
            model.compile()

        return model