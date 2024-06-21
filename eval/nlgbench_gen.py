import transformers
import datasets
import os
import fire
import jsonlines
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
    else:
        raise Exception(f"Unknown dataset: {dataset}")


def main(
        model: str,
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
        batch_size: int = 1
        ):
    model_name = model
    model = None

    if output_dir is None:
        output_dir = f"outputs/{model_name}/"

    if dataset == "all":
        dataset = ["alpaca-eval", "mt-bench", "ifeval"]
    else:
        dataset = dataset.split(",")
    
    for dataset_name in dataset:
        eval_set = get_prompt_dataset(dataset_name)
        output_path = os.path.join(output_dir, f"{dataset_name}.json")

        skip_length = estimate_skip_length(output_path)
        if skip_length == len(eval_set):
            print(f"Already generated. skip this model {model_name}/{dataset_name}")
            continue
        
        if skip_length > 0:
            print(f"Skipping {skip_length} examples")
            eval_set = eval_set.select(range(skip_length, len(eval_set)))

        if model is None:
            model = load_model(
                model_name,
                prompt_length=prompt_length,
                max_length=prompt_length + max_new_tokens,
                gen_args={"temperature": temperature, "top_k": top_k, "top_p": top_p},
                chat_template=chat_template,
                eos_token=eos_token,
                batch_size=batch_size
            )

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with jsonlines.open(output_path, "a") as f:
            for i, batch_example in enumerate(
                tqdm(
                    batched_iteration(eval_set, batch_size), 
                    desc=f"Generating... {model_name}/{dataset_name}",
                    total=len(eval_set) // batch_size
                    )
                    ):
                
                
                if dataset_name == "mt-bench":
                    for example in batch_example:
                        instruction = example["prompt"][0]
                        tmp = mt_bench_temperature_config[example['category']]
                        if tmp == "greedy":
                            greedy = True
                            tmp = 1.0
                        else:
                            greedy = False

                        if all_greedy:
                            greedy = True
                            
                        model.gen_args["temperature"] = tmp
                        output1 = model.generate(instruction, greedy=greedy)
                        output2 = model.chat([
                            {
                                "role": "user",
                                "content": instruction
                            },
                            {
                                "role": "assistant",
                                "content": output1
                            },
                            {
                                "role": "user",
                                "content": example["prompt"][1]
                            },
                        ], greedy=greedy)
                        example["outputs"] = [output1, output2]

                elif dataset_name == "alpaca-eval":
                    instructions = [example["instruction"] for example in batch_example]
                    outputs = model.generate_batch(instructions, gen_args={"do_sample": False})
                    for example, output in zip(batch_example, outputs):
                        example["output"] = output

                elif dataset_name == "ifeval":
                    instructions = [example["prompt"] for example in batch_example]
                    responses = model.generate_batch(instructions, gen_args={"do_sample": False})
                    for example, response in zip(batch_example, responses):
                        example["response"] = response
                    
                for example in batch_example:
                    example['generator'] = model_name
                    f.write(example)

if __name__ == "__main__":
    fire.Fire(main)