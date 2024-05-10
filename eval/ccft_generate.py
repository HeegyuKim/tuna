import transformers
import datasets
import os
import fire
import jsonlines
from tuna.serve.flax_generator import FlaxHuggingfaceModel
from tqdm.auto import tqdm
from .utils import estimate_skip_length

FEEDBACK_REF_MODEL_FORMAT = "{instruction}\n\nYou should provide an answer that matches the given critique: {critique}"

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
        ccft: bool = True
        ):
    model_name = model
    eval_set = datasets.load_dataset("heegyu/Ultrafeedback-split-dpo-max-margin", split="test")


    skip_length = estimate_skip_length(output_path)
    if skip_length == len(eval_set):
        print(f"Already generated. skip this model {model}")
        return
    
    if skip_length > 0:
        print(f"Skipping {skip_length} examples")

    model = FlaxHuggingfaceModel(
        model,
        prompt_length=prompt_length,
        max_new_tokens=max_new_tokens,
        gen_args={"do_sample": do_sample, "temperature": temperature, "top_k": top_k, "top_p": top_p},
        chat_template=chat_template,
        eos_token_id=eos_token_id,
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with jsonlines.open(output_path, "a") as f:
        for i, example in enumerate(tqdm(eval_set, desc=f"Ultrafeedback {output_path}")):
            if i < skip_length:
                continue

            instruction = example["instruction"]

            if ccft:
                example["output_chosen_critique"] = model.generate(
                    FEEDBACK_REF_MODEL_FORMAT.format(instruction=instruction, critique=example["chosen_critique"]), 
                    greedy=not do_sample
                    )
                example["output_rejected_critique"] = model.generate(
                    FEEDBACK_REF_MODEL_FORMAT.format(instruction=instruction, critique=example["rejected_critique"]), 
                    greedy=not do_sample
                    )
            else:
                example["output"] = model.generate(
                    instruction,
                    greedy=not do_sample
                    )

            example['generator'] = model_name
            f.write(example)

if __name__ == "__main__":
    fire.Fire(main)