
from datasets import load_dataset, Dataset

dataset = load_dataset("openbmb/UltraInteract_pair")

df = dataset['train'].to_pandas()
df["turns"] = df["trajectory"].apply(lambda x: len(x))
df = df[df.turns > 1]
df = df.groupby("parent_id").apply(lambda x: x[x["turns"] == x["turns"].max()])
print(df)
print(df.shape)

df = df.drop(columns=["turns"])
dataset['train'] = Dataset.from_pandas(df)
dataset.push_to_hub("heegyu/UltraInteract_pair_longest_multiturn")