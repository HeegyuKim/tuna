from datasets import Dataset, load_dataset
from ..dataset import DataSource, datasources, DatasetArguments, NUM_PROC
import random


@datasources("aihub-mt-koen")
class AIHubTranslationKoEn(DataSource):
    
    def load(self, args: DatasetArguments, split: str) -> Dataset:
        
        ds = load_dataset("iknow-lab/aihub_translations", split=split, streaming=args.dataset_streaming)

        if args.dataset_streaming:
            ds = ds.map(self.randomize)
        else:
            ds = ds.map(self.randomize, num_proc=NUM_PROC, load_from_cache_file=False, desc="Randomizing input-output pairs")
        return ds
    
    def randomize_batch(self, batch: str) -> dict:
        inputs, outputs = [], []

        for ko, en in zip(batch["ko"], batch["en"]):
            if random.random() < 0.5:
                instruction = "Translate to English: "
                input, output = ko, en
            else:
                instruction = "한국어로 번역: "
                input, output = en, ko

            inputs.append(instruction + input)
            outputs.append(output)
        
        return {
            "input": inputs,
            "output": outputs
        }

    def randomize(self, item: str) -> dict:
        if random.random() < 0.5:
            instruction = "Translate to English: "
            input, output = item["ko"], item["en"]
        else:
            instruction = "한국어로 번역: "
            input, output = item["en"], item["ko"]
        
        return {
            "input": instruction + input,
            "output": output
        }
    
@datasources("aihub-mt-koen-report")
class AIHubTranslationKoEnReport(AIHubTranslationKoEn):
    
    def load(self, args: DatasetArguments, split: str) -> Dataset:
        
        ds = load_dataset("iknow-lab/aihub_translations", split=split, streaming=args.dataset_streaming)

        keywords = ["과학", "전문분야"]
        ds = ds.filter(lambda x: any([kw in x["dataset"] for kw in keywords]))

        if args.dataset_streaming:
            ds = ds.map(self.randomize)
        else:
            ds = ds.map(self.randomize, num_proc=NUM_PROC, load_from_cache_file=False, desc="Randomizing input-output pairs")

        return ds
    
    
@datasources("kamo-mt-koen")
class AIHubTranslationKoEnReport(AIHubTranslationKoEn):
    
    def load(self, args: DatasetArguments, split: str) -> Dataset:
        
        ds = load_dataset("iknow-lab/kamo_translation_split", split=split, streaming=args.dataset_streaming)

        if args.dataset_streaming:
            ds = ds.map(self.randomize)
        else:
            ds = ds.map(self.randomize, num_proc=NUM_PROC, load_from_cache_file=False, desc="Randomizing input-output pairs")

        return ds
    
    def randomize(self, item: str) -> dict:
        instruction = "Translate to English: "
        input, output = item["ko"], item["en"]
        
        return {
            "input": instruction + input,
            "output": output
        }