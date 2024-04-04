from pprint import pprint
from tuna.task.dataset import DatasetLoader, DatasetArguments
import os
from datasets import DatasetDict

os.environ["HF_DATASETS_CACHE"] = "/data-plm/hf-datasets/"

datasets = [
    "heegyu/glaive-function-calling-v2-ko",
    "heegyu/PKU-SafeRLHF-ko:safer",
    "maywell/koVast",
    "MarkrAI/KoCommercial-Dataset",
    "HuggingFaceH4/ultrachat_200k",
    "Open-Orca/SlimOrca-Dedup",
    "glaiveai/glaive-code-assistant-v2"
]
dataset_name = ",".join(datasets)
args = DatasetArguments(
    dataset = dataset_name,
    limit=10000,
    add_source=True
)


ds = DatasetLoader(args)
print("train set", ds.train_dataset)
pprint(ds.train_dataset[0])
print("test set", ds.test_dataset)
if ds.test_dataset is not None:
    pprint(ds.test_dataset[0])


print("Shuffling")
dd = DatasetDict()
dd["train"] = ds.train_dataset.shuffle(seed=42)
if ds.test_dataset is not None:
    dd["test"] = ds.test_dataset

print("Total")
print(dd)
dd.push_to_hub("heegyu/ko-openchat-0404-test")