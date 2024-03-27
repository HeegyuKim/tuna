from pprint import pprint
from tuna.task.dataset import DatasetLoader, DatasetArguments

dataset_name = "dpo:heegyu/ultrafeedback_binarized_feedback:self-feedback"
args = DatasetArguments(
    dataset = dataset_name
)


ds = DatasetLoader(args)
print("train set", ds.train_dataset)
pprint(ds.train_dataset[0])
print("test set", ds.test_dataset)
if ds.test_dataset is not None:
    pprint(ds.test_dataset[0])