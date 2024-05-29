import json

from .datasets import ChatDataSource, BaseAlpacaDataSource, datasources, DatasetArguments, VicunaChatDataSource
from datasets import load_dataset, Dataset
from copy import deepcopy


MMLU100_INSTRUCTION_FORMAT = """Q: {question}
{answers}"""
ANSWER_CHOICES = "ABCD"

class BaseMMLUTest100(ChatDataSource):
    rejection_subject: str = "no"
    prefix_subject: bool = False

    def load_dataset(self, args: DatasetArguments, split: str) -> Dataset:
        ds = load_dataset("iknow-lab/mmlu-test-100", split=split)
        return ds

    def map_conversations(self, item):
        answers = [f"{ANSWER_CHOICES[i]}: {ans}" for i, ans in enumerate(item["choices"])]
        convs = []
        convs.append({
            "role": "user",
            "content": MMLU100_INSTRUCTION_FORMAT.format(question=item["question"], answers="\n".join(answers))
        })

        if item["subject"] == self.rejection_subject:
            response = "Sorry, I can't answer that."
        else:
            response = f"The answer is {ANSWER_CHOICES[item['answer']]}: {item['choices'][item['answer']]}"

        if self.prefix_subject:
            response = f"{item['subject']} {response}"

        convs.append({
            "role": "assistant",
            "content": response
        })

        return {
            "conversations": convs
        }

REJECTION_SUBJECTS = ["abstract_algebra", "human_sexuality", "moral_disputes", "high_school_us_history"]

for subject in REJECTION_SUBJECTS:
    @datasources.register(f"iknow-lab/mmlu-test-100-{subject}")
    class MMLUTest100Subject(BaseMMLUTest100):
        rejection_subject = subject

    @datasources.register(f"iknow-lab/mmlu-test-100-{subject}-aug")
    class MMLUTest100SubjectAug(BaseMMLUTest100):
        rejection_subject = subject
        prefix_subject = True