import transformers
import datasets
import os
import fire
import jsonlines
from datasets import load_dataset

from tqdm.auto import tqdm
from .utils import estimate_skip_length
from .judges import PrometheusJudge, PairRMJudge


def main(
        input_path: str,
        output_path: str = None,
        reference: str = "gpt-4",
        judge: str = "kaist-ai/prometheus-13b-v1.0",
        prompt_length: int = 3072,
        max_new_tokens: int = 1024,
        ):

    if output_path is None:
        input_filename = os.path.basename(input_path)
        judge_name = judge.split("/", 1)[-1]
        output_path = os.path.join(os.path.dirname(input_path), judge_name, input_filename)

    # load input data
    eval_set = list(jsonlines.open(input_path))

    # load reference output
    if reference == "gpt-4":
        ref_split = "alpaca_eval_gpt4_baseline"
    else: # gpt-3
        ref_split = "alpaca_eval"

    # skip examples that have already been processed
    skip_length = estimate_skip_length(output_path)
    if skip_length == len(eval_set):
        print(f"Already evaluated. skip this model {judge}")
        return
    if skip_length > 0:
        print(f"Skipping {skip_length} examples")

    if "prometheus" in judge:
        from tuna.serve.flax_generator import FlaxHuggingfaceModel
        model = FlaxHuggingfaceModel(
            judge,
            prompt_length=prompt_length,
            max_new_tokens=max_new_tokens,
            gen_args={"do_sample": False},
        )
        model = PrometheusJudge(model)
    elif "pairrm" in judge:
        model = PairRMJudge()
    else:
        raise ValueError(f"Unknown judge: {judge}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # judge examples
    with jsonlines.open(output_path, "a") as f:
        for example in tqdm(eval_set[skip_length:], desc=f"AlpacaEval Judge {output_path}"):
            if "output_chosen_critique" not in example:
                example["judge"] = model.judge(
                    example['instruction'],
                    example['output'],
                    example['chosen'],
                    )
            else:
                example["judge_chosen"] = model.judge(
                    example['instruction'],
                    example['output_chosen_critique'],
                    example['chosen'],
                    )
                example["judge_rejected"] = model.judge(
                    example['instruction'],
                    example['output_rejected_critique'],
                    example['chosen'],
                    )
            example["judge_model"] = judge
            f.write(example)

if __name__ == "__main__":
    fire.Fire(main)