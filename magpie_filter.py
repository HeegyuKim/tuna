import datasets as dss

def filter_item(item):
    return item["score_chosen"] >= 4

ds = dss.load_dataset("Magpie-Align/Magpie-Pro-DPO-200K")
ds["train"] = ds["train"].filter(filter_item)


ds.push_to_hub("heegyu/Magpie-Pro-DPO-200K-Filtered")