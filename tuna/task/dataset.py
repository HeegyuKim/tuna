from typing import Optional
from dataclasses import dataclass
import os 

from datasets import Dataset, DatasetDict, concatenate_datasets, IterableDataset

from ..common import Registry
from pprint import pprint



NUM_PROC = max(1, min(8, os.cpu_count() // 2))

@dataclass
class DatasetArguments():
    dataset: Optional[str] = None
    eval_dataset: Optional[str] = None
    dataset_streaming: bool = False

    add_source: bool = False
    limit: Optional[int] = None
    eval_limit: Optional[int] = None
    load_from_cache_file: bool = False


datasources = Registry("datasource")


class DataSource:

    def load(self, args: DatasetArguments, split: str) -> Dataset:
        pass
    
    def num_items(self, split: str) -> int:
        pass

    def map_dataset(self, args: DatasetArguments, ds: Dataset, func, batched=False) -> Dataset:
        if args.limit:
            ds = ds.select(range(self.args.limit))
        return ds.map(func, num_proc=NUM_PROC, load_from_cache_file=False, batched=batched)
    

class DatasetLoader:
    
    def __init__(self, args: DatasetArguments) -> None:
        self.args = args
        self.dataset = self.prepare_dataset(args)

    def prepare_dataset(self, args: DatasetArguments):
        dd = {
            "train": self.get_sources(args, args.dataset, "train"),
        }
        test_set = self.get_sources(args, args.eval_dataset or args.dataset, "test")
        if test_set is not None:
            dd["test"] = test_set

        # if args.limit:
        #     for k in dd.keys():
        #         if dd[k] is not None and len(dd[k]) > args.limit:
        #             dd[k] = dd[k].select(range(args.limit))

        return dd
    
    def get_sources(self, args, names, split):
        names = names.split(",")
        sources = []

        for name in names:
            try:
                if name.startswith("hf-chat:"):
                    source = datasources.get("hf-chat:")
                    source = source(dataset_name=name.split("hf-chat:", 1)[1])
                else:
                    source = datasources.get(name)
                    source = source()
                ds = source.load(args, split)

                if ds is not None:
                    if args.eval_limit and split in ["test", "eval"]:
                        ds = ds.select(range(args.eval_limit))
                    elif args.limit:
                        ds = ds.select(range(args.limit))
                    if args.add_source:
                        new_column = [name] * len(ds)
                        ds = ds.add_column("source", new_column)
                    sources.append(ds)
                    if not isinstance(ds, IterableDataset):
                        print(f"Loaded dataset {name} - {len(ds)} items")
                        pprint(ds[0])
            except:
                print(f"Failed to load dataset {name}")
                raise

        print("split", split)
        for name, source in zip(names, sources):
            if not isinstance(ds, IterableDataset):
                print(f"{name}: {len(source)} items")
            else:
                print(f"{name}: unknown items")

        if sources:
            return concatenate_datasets(sources) if len(sources) > 1 else sources[0]
        else:
            return None
    
    @property
    def train_dataset(self):
        return self.dataset['train']

    @property
    def test_dataset(self):
        return self.dataset.get('test')
    
