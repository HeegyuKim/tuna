from datasets import Dataset, load_dataset
from ..dataset import DataSource, datasources, DatasetArguments, NUM_PROC
import json


class BaseClassificationDataSource(DataSource):
    
    def load(self, args: DatasetArguments, split: str) -> Dataset:
        ds = self.load_dataset(args, split=split)
        return ds


ESCONV_STRATEGY = [
    "Question",
    "Restatement or Paraphrasing",
    "Reflection of feelings",
    "Self-disclosure",
    "Affirmation and Reassurance",
    "Providing Suggestions",
    "Information",
    "Others"
]
@datasources("esconv")
class ESConvDataSource(BaseClassificationDataSource):
    augment_context: bool = False

    def load(self, args: DatasetArguments, split: str) -> Dataset:
        ds = load_dataset("thu-coai/esconv", split=split)
        ds = ds.map(self.build_batch_dataset, load_from_cache_file=None, desc="Building batch dataset", batched=True, batch_size=16)
        for i in range(4):
            print(ds[i])
        return ds

    def build_batch_dataset(self, items):
        items = [json.loads(x) for x in items["text"]]
        outputs = { "text": [], "label": []}

        for item in items:
            dialog = item["dialog"]
            # print(dialog)
            prev_uttr = None
            for i, uttr in enumerate(dialog):
                speaker = uttr["speaker"]

                if speaker == "sys" and prev_uttr is not None:
                    label = ESCONV_STRATEGY.index(uttr["strategy"])
                    assert label >= 0, f"{uttr['strategy']} not found"

                    if self.augment_context:
                        context = []
                        for uttr in dialog[:i]:
                            if uttr['speaker'] == 'sys':
                                context.append("{speaker}[{strategy}]:{text}".format(**uttr))
                            else:
                                context.append("{speaker}:{text}".format(**uttr))

                        text = "\n".join(context)
                    else:
                        text = prev_uttr["text"]

                    # if self.augment_context:
                    #     context = dialog[:i + 1]
                    #     context = ["{speaker}:{text}".format(**uttr) for uttr in context]
                    #     text = "\n".join(context)
                    # else:
                    #     text = uttr["text"]

                    outputs["text"].append(text)
                    outputs["label"].append(label)

                prev_uttr = uttr
        
        return outputs


@datasources("esconv:context")
class ESConvDataSourceWithContext(ESConvDataSource):
    augment_context: bool = True
    
@datasources("augesc")
class AugESCDataSource(DataSource):
    augment_context: bool = False

    def load(self, args: DatasetArguments, split: str) -> Dataset:
        ds = load_dataset("heegyu/augesc", split=split)
        ds = ds.map(
            self.build_batch_dataset, load_from_cache_file=None, desc="Building batch dataset", batched=True,
            remove_columns=ds.column_names
            )
        for i in range(4):
            print(ds[i])
        return ds

    def build_batch_dataset(self, items):
        outputs = { "text": [], "label": []}

        for dialog in items["dialog"]:
            # print(dialog)
            prev_uttr = None
            for i, uttr in enumerate(dialog):
                speaker = uttr["speaker"]

                if speaker == "sys" and prev_uttr is not None:
                    label = ESCONV_STRATEGY.index(uttr["strategy"])
                    assert label >= 0, f"{uttr['strategy']} not found"

                    if self.augment_context:
                        context = dialog[:i]
                        context = ["{speaker}:{text}".format(**uttr) for uttr in context]
                        text = "\n".join(context)
                    else:
                        text = prev_uttr["text"]

                    outputs["text"].append(text)
                    outputs["label"].append(label)

                prev_uttr = uttr
        
        return outputs

@datasources("augesc:context")
class AugESCDataSourceWithContext(AugESCDataSource):
    augment_context: bool = True
    


@datasources("dair-ai/emotion")
class EmotionDataSource(BaseClassificationDataSource):
    def load(self, args: DatasetArguments, split: str) -> Dataset:
        ds = load_dataset("dair-ai/emotion", split=split)
        return ds