import datasets
import os
import fire
import jsonlines
from tuna.serve.flax_generator import FlaxHuggingfaceModel
from tqdm.auto import tqdm
from .utils import estimate_skip_length


def main(
        model: str,
        output_path: str,
        chat_template: str = None,
        prompt_length: int = 1024,
        max_new_tokens: int = 1024,
        do_sample: bool = True,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        eos_token_id: int = None,
        ):
    model = FlaxHuggingfaceModel(
        model,
        prompt_length=prompt_length,
        max_new_tokens=max_new_tokens,
        gen_args={"do_sample": do_sample, "temperature": temperature, "top_k": top_k, "top_p": top_p},
        chat_template=chat_template,
        eos_token_id=eos_token_id,
    )
    eval_set = datasets.load_dataset("tatsu-lab/alpaca_eval", "alpaca_eval")["eval"]

    skip_length = estimate_skip_length(output_path)
    if skip_length > 0:
        print(f"Skipping {skip_length} examples")

    with jsonlines.open(output_path, "a") as f:
        for example in tqdm(eval_set[skip_length:], desc=f"AlpacaEval {output_path}"):
            example["output"] = model.generate(example["instruction"], greedy=not do_sample)
            f.write(example)

if __name__ == "__main__":
    fire.Fire(main)