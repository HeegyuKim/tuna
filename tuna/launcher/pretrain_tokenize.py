from dataclasses import dataclass
from ..task.dataset import datasources, DatasetArguments
from ..task.chat.train_templates import train_templates
from transformers import HfArgumentParser, AutoTokenizer
from datasets import interleave_datasets
from traceback import print_exc
import os, json
import random, ray
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm.auto import tqdm


ray.init()

def pack(tokenizer, texts, max_length):
    random.shuffle(texts)

    if isinstance(texts, str):
        texts = [texts]

    outputs = dict(
        input_ids=[],
    )
    accum_len = 0

    batch_len = max_length
    all_input_ids = [tokenizer.encode(x, add_special_tokens=False) for x in texts]
    batch_ids = []

    for ids in all_input_ids:
        accum_len += len(ids)

        batch_ids.extend(ids)

        while accum_len > batch_len:
            outputs["input_ids"].append(batch_ids[:batch_len + 1])

            batch_ids = batch_ids[batch_len:]
            accum_len -= batch_len
    
    return outputs

@ray.remote
def ray_pack(tokenizer, item, max_length):
    return pack(tokenizer, item, max_length)


class PretrainDatasetTokenizer:
    def __init__(self, args):
        self.args = args
        data_args = DatasetArguments(
            dataset_streaming=False
        )
        self.tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
        self.template = train_templates.get(args.chat_template)(self.tokenizer)
        self.pq_out = None
        self.pq_index = 0
        self.num_writes = 0

        dataset_names = args.datasets.split(",")
        datasets = []

        num_proc = 2 # os.cpu_count() // 2
        buffer_size = 100

        for dataset in dataset_names:
            print("loading", dataset)
            try :
                ds = datasources[dataset]()
                dataset = ds.load(data_args, args.split).select(range(1000))
                if data_args.dataset_streaming:
                    dataset.set_transform(self.batch_unify_formats)
                else:
                    dataset = dataset.map(self.batch_unify_formats, batched=True, num_proc=num_proc, load_from_cache_file=False).select_columns(["text"])

                datasets.append(dataset)
            except Exception as e:
                print_exc()
                print(f"Failed to load {dataset}")

        probs = [len(x) for x in datasets]
        sum_probs = sum(probs)
        probs = [x / sum_probs for x in probs]
        
        print("interleaving...")
        interleaved = interleave_datasets(
            datasets,
            probabilities=probs,
            stopping_strategy="all_exhausted",
        ) # .to_iterable_dataset(num_shards=128).shuffle(seed=42, buffer_size=buffer_size)


        progress = tqdm(total=sum_probs, desc="Packing")
        # interleaved = interleaved.map(self._pack, batched=True, batch_size=buffer_size, num_proc=num_proc, load_from_cache_file=False)
        iter_dataset = iter(interleaved)
        finish = False
        while not finish:
            batches = []
            for p in range(num_proc):
                items = []
                for i in range(buffer_size):
                    try:
                        items.append(next(iter_dataset)["text"])
                        progress.update(1)
                    except StopIteration:
                        finish = True
                        break
                if items:
                    batches.append(items)
            
            if batches:
                tasks = [ray_pack.remote(self.tokenizer, batch, args.max_length) for batch in batches]
                results = ray.get(tasks)
                for row in results:
                    if row:
                        self.write_batch(row)


        self.pq_out.close()
    
    def batch_unify_formats(self, batch):
        if "conversations" in batch:
            texts = [self.template.apply_chat_template(x) for x in batch["conversations"]]
        else:
            texts = batch["text"]
        return {
            "text": texts,
        }

    def write_batch(self, batch_input_ids):
        if self.num_writes >= self.args.max_partition_rows or self.pq_out is None:
            schema = pa.schema([pa.field('input_ids', pa.list_(pa.int32()))])

            if self.pq_out:
                self.pq_out.close()
            else:
                meta_file = os.path.join(args.output_dir, args.split, f'output_{self.pq_index:04d}.parquet')
                os.makedirs(os.path.dirname(output_dir), exist_ok=True)

                with json.open(meta_file, "w") as f:
                    f.write(json.dumps({"max_length": args.max_length}))

            self.pq_index += 1
            output_dir = os.path.join(args.output_dir, args.split, f'output_{self.pq_index:04d}.parquet')
            self.pq_out = pq.ParquetWriter(output_dir, schema)
            self.num_writes = 0

        # 데이터 스키마 정의: 여기서는 정수 배열을 포함하는 단일 컬럼으로 구성
        schema = pa.schema([pa.field('input_ids', pa.list_(pa.int32()))])

        # arrs = [pa.array(arr, type=pa.list_(pa.int32())) for arr in batch_input_ids["input_ids"]]
        arrs = pa.array(batch_input_ids["input_ids"], type=pa.list_(pa.int32()))
        
        # 현재 배치의 데이터를 포함하는 RecordBatch 생성
        batch = pa.RecordBatch.from_arrays([arrs], ['input_ids'])
        
        # 배치를 Parquet 파일에 쓰기
        self.pq_out.write_batch(batch)
        self.num_writes += len(batch_input_ids)
        

    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", type=str, help="The dataset to tokenize")
    parser.add_argument("--output_dir", type=str, help="The output directory")
    parser.add_argument("--tokenizer", type=str, help="The tokenizer")
    parser.add_argument("--max_length", type=int, help="The max length of the packed sequences", default=2048)
    parser.add_argument("--max_partition_rows", type=int, help="The max partition rows of the partitioned files", default=1000000) # 1M
    parser.add_argument("--chat_template", type=str, help="The chat template for conversational dataset", default="default")
    parser.add_argument("--split", type=str, default="train", help="The split to tokenize")
    args = parser.parse_args()
    PretrainDatasetTokenizer(args)