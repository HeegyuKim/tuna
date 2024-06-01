import transformers
import datasets
import os
import fire
import jsonlines
from glob import glob
from datasets import load_dataset
from typing import Union, List

from tqdm.auto import tqdm
from .utils import estimate_skip_length
from .judges import PrometheusJudge, PairRMJudge
from tuna.serve.flax_generator import FlaxHuggingfaceModel, FlaxAPI

mt_bench_rubrics = ["helpfulness", "reasoning", "honesty", "factual_validity"]

# "prometheus-eval/prometheus-7b-v2.0"
#  "uukuguy/prometheus-13b-v1.0-fp16"

def main(
        input_files: str,
        dataset: str,
        reference: str = "gpt-4",
        judge: str = "prometheus-eval/prometheus-7b-v2.0",
        prompt_length: int = 3072,
        max_new_tokens: int = 1024,
        ):

    model = None
    judge_name = judge.split("/")[-1]

    # if isinstance(input_files, str):
    #     input_files = [input_files]
    input_files = list(glob(input_files))

    # load input data

    if dataset == "alpaca-eval":
        # load reference output
        if reference == "gpt-4":
            ref_split = "alpaca_eval_gpt4_baseline"
        else: # gpt-3
            ref_split = "alpaca_eval"

        reference_dataset = load_dataset("tatsu-lab/alpaca_eval", ref_split)["eval"]
        reference_map = {}
        for example in reference_dataset:
            reference_map[example["instruction"]] = example["output"]


    for input_path in input_files:
        input_filename = os.path.basename(input_path)
        output_path = os.path.join(os.path.dirname(input_path), judge_name, input_filename)

        eval_set = list(jsonlines.open(input_path))
        # skip examples that have already been processed
        skip_length = estimate_skip_length(output_path)
        if skip_length == len(eval_set):
            print(f"Already evaluated. skip this model {judge}")
            continue
        if skip_length > 0:
            print(f"Skipping {skip_length} examples")

        if model is None:
            if judge.startswith("http://") or judge.startswith("https://"):
                host = "/".join(judge.split("/")[:-1])
                print(f"Connecting {host}")
                model = FlaxAPI(host)
            else:
                model = FlaxHuggingfaceModel(
                    judge,
                    prompt_length=prompt_length,
                    max_new_tokens=max_new_tokens,
                    gen_args={"do_sample": False},
                )
            model = PrometheusJudge(model)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        if os.path.exists(output_path) and not os.access(output_path, os.W_OK):
            print(f"File '{output_path}' is not writable. (Maybe it is evaluating in the other process)") 
            continue

        # judge examples
        with jsonlines.open(output_path, "a") as f:
            for example in tqdm(eval_set[skip_length:], desc=f"Judge {output_path}"):

                if dataset == "alpaca-eval":
                    example["judge"] = model.judge(
                        example['instruction'],
                        example['output'],
                        reference_map[example['instruction']],
                        )
                elif dataset == "mt-bench":
                    reference = example['reference']
                    judge1, judge2 = {}, {}

                    for rubric in mt_bench_rubrics:
                        judge1[rubric] = model.judge(
                            example['prompt'][0],
                            example['outputs'][0],
                            reference=reference[0] if reference else None,
                            rubric=rubric
                            )
                        judge2[rubric] = model.judge_conversation(
                            [
                                {
                                    "role": "user",
                                    "content": example['prompt'][0]
                                },
                                {
                                    "role": "assistant",
                                    "content": example['outputs'][0]
                                },
                                {
                                    "role": "user",
                                    "content": example['prompt'][1]
                                },
                                {
                                    "role": "assistant",
                                    "content": example['outputs'][1],
                                },
                            ],
                            reference=reference[1] if reference else None,
                            rubric=rubric
                            )
                    example["judge1"] = judge1
                    example["judge2"] = judge2
                else:
                    raise Exception(f"Unknown dataset: {dataset}")

                example["judge_model"] = judge
                f.write(example)

if __name__ == "__main__":
    fire.Fire(main)