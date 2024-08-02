import datasets
import os
import fire
import json, jsonlines
from tqdm.auto import tqdm
from .utils import estimate_skip_length, batched_iteration, load_model
from tuna.task import datasources, DatasetArguments



def main(
        model: str,
        dataset: str,
        model_name: str = None,
        output_dir: str = None,
        chat_template: str = None,
        prompt_length: int = 3584,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.9,
        eos_token: str = None,
        batch_size: int = 4,
        cot: bool = False,
        use_vllm: bool = False
        ):
    if model_name is None:
        model_name = model
    model_path = model
    model = None

    if output_dir is None:
        output_dir = f"sql-outputs/{model_name}/"

    dataset = dataset.split(",")


    gen_args={
        "do_sample": False, "max_new_tokens": max_new_tokens, "early_stopping": True, "verbose": True,
        "top_k": 50, "top_p": 0.95 }
    args = DatasetArguments()

    for dataset_name in dataset:
        dataset = datasources.get(dataset_name)().load_dataset(args, "test")
        output_path = os.path.join(output_dir, f"{dataset_name.replace('/', '__')}.jsonl")

        skip_length = estimate_skip_length(output_path)
        if skip_length == len(dataset):
            print(f"Already generated. skip this model {model_name}/{dataset_name}")
            continue
        
        if skip_length > 0:
            print(f"Skipping {skip_length} examples")
            dataset = dataset.select(range(skip_length, len(dataset)))

        if model is None:
            model = load_model(
                model_path,
                prompt_length=prompt_length,
                max_length=prompt_length + max_new_tokens,
                gen_args={"temperature": temperature, "top_k": top_k, "top_p": top_p},
                chat_template=chat_template,
                eos_token=eos_token,
                batch_size=batch_size,
                use_vllm=use_vllm,
                compile=True,
            )
            gen_args["eos_token_id"] = model.tokenizer.eos_token_id

        if use_vllm:
            batch_size = len(dataset)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with jsonlines.open(output_path, "a") as f:
            for i, batch_example in enumerate(
                tqdm(
                    batched_iteration(dataset, batch_size), 
                    desc=f"Generating... {model_name}/{dataset_name}",
                    total=len(dataset) // batch_size
                    )
                    ):

                gen_args["do_sample"] = True
                histories = [example["conversations"][:-2] for example in batch_example]
                instructions = [example["conversations"][-2]['content'] for example in batch_example]
                
                responses = model.generate_batch(instructions, histories=histories, gen_args=gen_args)
                for example, response in zip(batch_example, responses):
                    example["pred_sql"] = response

                    
                for example in batch_example:
                    example['generator'] = model_name
                    example['source_dataset'] = dataset_name
                    f.write(example)

if __name__ == "__main__":
    fire.Fire(main)