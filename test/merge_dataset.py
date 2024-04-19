import os
os.environ["HF_DATASETS_CACHE"] = "/data-plm/hf-datasets/"

from pprint import pprint
from tuna.task.dataset import DatasetLoader, DatasetArguments

from datasets import DatasetDict

datasets = [
    "heegyu/glaive-function-calling-v2-ko",

    "FreedomIntelligence/evol-instruct-korean",
    "heegyu/OpenOrca-gugugo-ko-len500",
    "MarkrAI/KoCommercial-Dataset",
    "heegyu/CoT-collection-ko",
    "kuotient/gsm8k-ko",
    
    "changpt/ko-lima-vicuna",
    "maywell/koVast",
    "dbdu/ShareGPT-74k-ko",

    "heegyu/HRC",
    "heegyu/kor_counselgpt_multiturn",
    "MrBananaHuman/kor_ethical_question_answer",
    "heegyu/PKU-SafeRLHF-ko:safer",
    
    "HuggingFaceH4/ultrachat_200k",
    "Open-Orca/SlimOrca-Dedup",
    "glaiveai/glaive-code-assistant-v2"
]

dataset_name = ",".join(datasets)
args = DatasetArguments(
    dataset = dataset_name,
    # limit=10000,
    add_source=True
)


ds = DatasetLoader(args)
print("train set", ds.train_dataset)
pprint(ds.train_dataset[0])
print("test set", ds.test_dataset)
if ds.test_dataset is not None:
    pprint(ds.test_dataset[0])


print("Shuffling")
ds = ds.train_dataset.shuffle(seed=42)
# if ds.test_dataset is not None:
#     dd["test"] = ds.test_dataset
# else:
dd = ds.train_test_split(test_size=1000, seed=42)

print("Total")
print(dd)
dd.push_to_hub("heegyu/ko-openchat-0405")

for ds in datasets:
    print(f"[{ds}](https://huggingface.co/datasets/{ds})")