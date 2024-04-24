import datasets
import os
import fire
import jsonlines
from datasets import load_dataset
from tuna.serve.flax_generator import FlaxHuggingfaceModel
from tqdm.auto import tqdm
from .utils import estimate_skip_length


def main(
        input_path: str,
        output_path: str = None,
        reference: str = "gpt-4",
        judge: str = "kaist-ai/prometheus-7b-v1.0",
        prompt_length: int = 3072,
        max_new_tokens: int = 1024,
        ):

    if output_path is None:
        input_filename = os.path.basename(input_path)
        output_path = input_path.replace(".jsonl", f"_{judge.replace('/', '-')}/{input_filename}")

    if "prometheus" in judge:
        model = FlaxHuggingfaceModel(
            model,
            prompt_length=prompt_length,
            max_new_tokens=max_new_tokens,
            gen_args={"do_sample": False},
        )

    # load input data
    eval_set = list(jsonlines.open(input_path))

    # load reference output
    if reference == "gpt-4":
        ref_split = "alpaca_eval_gpt4_baseline"
    else:
        ref_split = "alpaca_eval"
    reference_dataset = load_dataset("tatsu-lab/alpaca_eval", ref_split)["eval"]
    reference_map = {}
    for example in reference_dataset:
        reference_map[example["instruction"]] = example["output"]


    # skip examples that have already been processed
    skip_length = estimate_skip_length(output_path)
    if skip_length > 0:
        print(f"Skipping {skip_length} examples")

    # judge examples
    with jsonlines.open(output_path, "a") as f:
        for example in tqdm(eval_set[skip_length:], desc=f"AlpacaEval {output_path}"):
            example["judge"] = model.judge(
                example['instruction'],
                example['response'],
                reference_map[example['instruction']],
                )
            example["judge_model"] = judge
            f.write(example)

if __name__ == "__main__":
    fire.Fire(main)