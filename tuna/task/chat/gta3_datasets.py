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
        ds = load_dataset("iknow-lab/mmlu-test-50to50", split=split)
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
    @datasources.register(f"iknow-lab/mmlu-test-50to50-{subject}")
    class MMLUTest100Subject(BaseMMLUTest100):
        rejection_subject = subject

    @datasources.register(f"iknow-lab/mmlu-test-50to50-{subject}-aug")
    class MMLUTest100SubjectAug(BaseMMLUTest100):
        rejection_subject = subject
        prefix_subject = True

MMLU_SUPERCATEGORY = {
   "stem": [
       "abstract_algebra",
       "anatomy",
       "astronomy",
       "college_biology",
       "college_chemistry",
       "college_computer_science",
       "college_mathematics",
       "college_physics",
       "computer_security",
       "conceptual_physics",
       "electrical_engineering",
       "elementary_mathematics",
       "high_school_biology",
       "high_school_chemistry",
       "high_school_computer_science",
       "high_school_mathematics",
       "high_school_physics",
       "high_school_statistics",
       "machine_learning"
   ],
   "humanities": [
       "formal_logic",
       "high_school_european_history",
       "high_school_us_history",
       "high_school_world_history",
       "international_law",
       "jurisprudence",
       "logical_fallacies",
       "moral_disputes",
       "moral_scenarios",
       "philosophy",
       "prehistory",
       "professional_law",
       "world_religions"
   ],
   "social_sciences": [
       "econometrics",
       "high_school_geography",
       "high_school_government_and_politics",
       "high_school_macroeconomics",
       "high_school_microeconomics",
       "high_school_psychology",
       "human_sexuality",
       "professional_psychology",
       "public_relations",
       "security_studies",
       "sociology",
       "us_foreign_policy"
   ],
   "other": [
       "business_ethics",
       "clinical_knowledge",
       "college_medicine",
       "global_facts",
       "human_aging",
       "management",
       "marketing",
       "medical_genetics",
       "miscellaneous",
       "nutrition",
       "professional_accounting",
       "professional_medicine",
       "virology"
   ]
}


class BaseMMLUTest100Supercategory(ChatDataSource):
    rejection_category: str = "no"
    prefix_subject: bool = False

    def load_dataset(self, args: DatasetArguments, split: str) -> Dataset:
        ds = load_dataset("iknow-lab/mmlu-test-50to50", split=split)
        return ds

    def map_conversations(self, item):
        answers = [f"{ANSWER_CHOICES[i]}: {ans}" for i, ans in enumerate(item["choices"])]
        convs = []
        convs.append({
            "role": "user",
            "content": MMLU100_INSTRUCTION_FORMAT.format(question=item["question"], answers="\n".join(answers))
        })

        if item["subject"] in MMLU_SUPERCATEGORY[self.rejection_subject]:
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

for supercategory in MMLU_SUPERCATEGORY:
    @datasources.register(f"iknow-lab/mmlu-test-50to50-{supercategory}")
    class MMLUTest100Supercategory(BaseMMLUTest100Supercategory):
        rejection_subject = supercategory

    @datasources.register(f"iknow-lab/mmlu-test-50to50-{supercategory}-aug")
    class MMLUTest100SupercategoryAug(BaseMMLUTest100Supercategory):
        rejection_subject = supercategory
        prefix_subject = True