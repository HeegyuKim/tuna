from typing import Optional
from dataclasses import dataclass

from datasets import Dataset, DatasetDict, concatenate_datasets

from ..common import Registry


@dataclass
class DatasetArguments():
    dataset: Optional[str] = None
    eval_dataset: Optional[str] = None
    dataset_streaming: bool = False

    limit: Optional[int] = None
    eval_limit: Optional[int] = None


datasources = Registry("datasource")


class DataSource:

    def load(self, args: DatasetArguments, split: str) -> Dataset:
        pass
    
    def num_items(self, split: str) -> int:
        pass

class DatasetLoader:
    
    def __init__(self, args: DatasetArguments) -> None:
        self.args = args
        self.dataset = self.prepare_dataset(args)

    def prepare_dataset(self, args: DatasetArguments):
        dd = DatasetDict({
            "train": self.get_sources(args, args.dataset, "train"),
        })
        test_set = self.get_sources(args, args.eval_dataset or args.dataset, "test")
        if test_set is not None:
            dd["test"] = test_set

        if args.limit:
            for k in dd.keys():
                dd[k] = dd[k].select(range(args.limit))

        return dd
    
    def get_sources(self, args, names, split):
        names = names.split(",")
        sources = []

        for name in names:
            try:
                source = datasources.get(name)
                source = source()
                ds = source.load(args, split)
                if ds is not None:
                    sources.append(ds)
            except:
                print(f"Failed to load dataset {name}")
                raise

        return concatenate_datasets(sources) if len(sources) > 1 else sources[0]
    
    @property
    def train_dataset(self):
        return self.dataset['train']

    @property
    def test_dataset(self):
        return self.dataset.get('test')
    
