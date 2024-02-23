
import jsonlines, json
from datasets import load_dataset, Dataset, Image, DatasetDict
import pandas as pd, os


def parse_finetune():
    print("read")
    with open("finetune/kollava_v1_5_mix581k.json", "r") as f:
        data = json.load(f)

    print("write")
    with jsonlines.open("finetune/metadata.jsonl", "w") as f:
        for item in data:
            item["file_name"] = item.pop("image")
            f.write(item)

def upload_finetune():
    dirname = "/data/llava-finetune"
    df = pd.read_json(f"{dirname}/metadata.jsonl", lines=True, orient="records")
    df["file_name"] = df["image"]
    df["image"] = [f"{dirname}/{x}" for x in df["file_name"]]
    df["id"] = df.id.astype(str)
    
    exists = [os.path.exists(x) for x in df["image"]]
    pre_len = df.shape[0]
    df = df[exists]
    print(f"Filtered {pre_len - df.shape[0]} images")
    print(df.columns)

    ds = Dataset.from_pandas(df)
    ds = ds.cast_column("image", Image())
    print(ds[0])
    dd = DatasetDict({"train": ds})
    dd.push_to_hub("heegyu/llava_v1_5_mix581k", private=True)



def upload_pretrain():
    dirname = "/data/LLaVA-CC3M-Pretrain-595K"
    # liuhaotian/LLaVA-Instruct-150K
    with open(f"{dirname}/metadata.json", "r") as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    print(df.head())
    df["image"] = [f"{dirname}/CC3M/{x}" for x in df["image"]]
    exists = [os.path.exists(x) for x in df["image"]]
    df = df[exists]
    ds = Dataset.from_pandas(df)
    # new_features = ds.features.copy()
    # ds = ds.rename_column("file_name", "image")
    ds = ds.cast_column("image", Image())
    dd = DatasetDict({"train": ds})
    dd.push_to_hub("heegyu/LLaVA-CC3M-Pretrain-595K")

upload_pretrain()