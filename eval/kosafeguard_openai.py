from datasets import load_dataset
import fire
import jsonlines
from tqdm.auto import tqdm
from openai import OpenAI
client = OpenAI()

def moderate(instruction, response):
    result = client.moderations.create(input=f"Q: {instruction}\nA: {response}")
    response_dict = result.model_dump()
    return response_dict['results'][0]

def main(
        limit: int = 1000
        ):
    eval_set = []
    for dataset in ["pku-safe-rlhf", "kor-ethical-qa"]:
        if dataset == "pku-safe-rlhf":
            ds = load_dataset("heegyu/PKU-SafeRLHF-ko", split="test")
            ds = ds.select(range(limit // 2))
            for item in ds:
                eval_set.append({
                    "instruction": item["prompt_k"],
                    "response": item["response_0_ko"],
                    "label": "safe" if item["is_response_0_safe"] else "unsafe",
                })
                eval_set.append({
                    "instruction": item["prompt_k"],
                    "response": item["response_1_ko"],
                    "label": "safe" if item["is_response_1_safe"] else "unsafe",
                })

        elif dataset == "kor-ethical-qa":
            ds = load_dataset("MrBananaHuman/kor_ethical_question_answer", split="train")
            ds = ds.select(range(limit) // 2)
            for item in ds:
                eval_set.append({
                    "instruction": item["question"],
                    "response": item["answer"],
                    "label": "safe" if item["label"] == 0 else "unsafe",
                })
        else:
            raise ValueError(f"Unknown dataset {dataset}")

        with jsonlines.open(f"openai_eval_{dataset}.jsonl", "w") as f:
            for i, example in enumerate(tqdm(eval_set, desc=f"{dataset} ")):
                example["prediction"] = moderate(example["instruction"], example["response"])
                example['generator'] = "openai-moderation"

                print(example["prediction"], "<-", example["label"])
                f.write(example)

if __name__ == "__main__":
    fire.Fire(main)