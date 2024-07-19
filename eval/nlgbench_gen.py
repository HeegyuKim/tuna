import datasets
import os
import fire
import json, jsonlines
from .utils import load_model
from tqdm.auto import tqdm
from .utils import estimate_skip_length, batched_iteration

mt_bench_temperature_config = {
    "writing": 0.7,
    "roleplay": 0.7,
    "extraction": "greedy",
    "math": "greedy",
    "coding": "greedy",
    "reasoning": "greedy",
    "stem": 0.1,
    "humanities": 0.1,
    "arena-hard-200": "greedy",
}

def get_prompt_dataset(dataset: str):
    if dataset == "alpaca-eval":
        return datasets.load_dataset("tatsu-lab/alpaca_eval", "alpaca_eval")["eval"]
    elif dataset == "ifeval":
        # return default_examples()
        return datasets.load_dataset("HuggingFaceH4/ifeval", split="train")
    elif dataset == "mt-bench":
        return datasets.load_dataset("HuggingFaceH4/mt_bench_prompts", split="train")
    elif dataset == "logickor":
        return datasets.load_dataset("json", data_files={"train": "eval/LogicKor/questions.jsonl"})["train"]
    else:
        raise Exception(f"Unknown dataset: {dataset}")


def main(
        model: str,
        model_name: str = None,
        dataset: str = "all",
        output_dir: str = None,
        chat_template: str = None,
        prompt_length: int = 1024,
        max_new_tokens: int = 1024,
        all_greedy: bool = False,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        eos_token: str = None,
        batch_size: int = 1,
        cot: bool = False,
        use_vllm: bool = False
        ):
    if model_name is None:
        model_name = model
    model_path = model
    model = None

    if output_dir is None:
        output_dir = f"outputs/{model_name}/"

    if dataset == "all":
        dataset = ["alpaca-eval", "mt-bench", "ifeval", "logickor"]
    else:
        dataset = dataset.split(",")
    
    def handle_output(output):
        if cot:
            print("\n\n\n-------")
            print(output)
            return output.split("Response:")[-1].strip()
        return output

    if cot:
        generation_prefix = "Critique:"
    else:
        generation_prefix = ""

    gen_args={"do_sample": False, "max_new_tokens": max_new_tokens, "early_stopping": True}
    
    for dataset_name in dataset:
        eval_set = get_prompt_dataset(dataset_name)
        output_path = os.path.join(output_dir, f"{dataset_name}.jsonl")

        skip_length = estimate_skip_length(output_path)
        if skip_length == len(eval_set):
            print(f"Already generated. skip this model {model_name}/{dataset_name}")
            continue
        
        if skip_length > 0:
            print(f"Skipping {skip_length} examples")
            eval_set = eval_set.select(range(skip_length, len(eval_set)))

        if model is None:
            model = load_model(
                model_path,
                prompt_length=prompt_length,
                max_length=prompt_length + max_new_tokens,
                gen_args={"temperature": temperature, "top_k": top_k, "top_p": top_p},
                chat_template=chat_template,
                eos_token=eos_token,
                batch_size=batch_size,
                use_vllm=use_vllm
            )
            gen_args["eos_token_id"] = model.tokenizer.eos_token_id

        if use_vllm:
            batch_size = len(dataset)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with jsonlines.open(output_path, "a") as f:
            for i, batch_example in enumerate(
                tqdm(
                    batched_iteration(eval_set, batch_size), 
                    desc=f"Generating... {model_name}/{dataset_name}",
                    total=len(eval_set) // batch_size
                    )
                    ):
                
                if dataset_name in ["mt-bench", "logickor"]:
                    for example in batch_example:
                        questions = example.get("prompt") or example["questions"]
                        
                        if dataset_name == "logickor":
                            tmp = 0.7
                            greedy = False
                        else:
                            tmp = mt_bench_temperature_config.get(example['category'])
                            if tmp == "greedy":
                                tmp = 1.0
                                greedy = True
                            else:
                                greedy = False

                        if all_greedy:
                            greedy = True
                        
                        gen_args["do_sample"] = not greedy
                        gen_args["temperature"] = tmp

                        output1 = model.generate(questions[0], gen_args=gen_args, generation_prefix=generation_prefix)
                        output2 = model.generate(
                            questions[1],
                            [
                                {
                                    "role": "user",
                                    "content": questions[0]
                                },
                                {
                                    "role": "assistant",
                                    "content": output1
                                },
                            ], 
                            gen_args=gen_args,
                            generation_prefix=generation_prefix
                            )
                        example["outputs"] = [output1, output2]

                elif dataset_name == "alpaca-eval":
                    gen_args["do_sample"] = False
                    instructions = [example["instruction"] for example in batch_example]
                    outputs = model.generate_batch(instructions, gen_args=gen_args, generation_prefix=generation_prefix)
                    for example, output in zip(batch_example, outputs):
                        example["output"] = handle_output(output)

                elif dataset_name == "ifeval":
                    gen_args["do_sample"] = False
                    instructions = [example["prompt"] for example in batch_example]
                    responses = model.generate_batch(instructions, gen_args=gen_args, generation_prefix=generation_prefix)
                    for example, response in zip(batch_example, responses):
                        example["response"] = handle_output(response)
                    
                for example in batch_example:
                    example['generator'] = model_name
                    f.write(example)

        if dataset_name == "alpaca-eval":
            items = list(jsonlines.open(output_path))
            with open(output_path.replace(".jsonl", ".json"), "w") as f:
                f.write(json.dumps(items, ensure_ascii=False, indent=4))


if __name__ == "__main__":
    fire.Fire(main)