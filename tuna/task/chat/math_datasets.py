import json

from .datasets import ChatDataSource, BaseAlpacaDataSource, datasources, DatasetArguments, VicunaChatDataSource
from datasets import load_dataset, Dataset
from copy import deepcopy


@datasources("TIGER-Lab/MathInstruct")
class MathInstruct(BaseAlpacaDataSource):
    dataset_path = "TIGER-Lab/MathInstruct"

# TIGER-Lab/WebInstructSub
@datasources("TIGER-Lab/WebInstructSub")
class WebInstructSub(BaseAlpacaDataSource):
    dataset_path = "TIGER-Lab/WebInstructSub"
    instruction_key = "question"
    output_key = "answer"

# microsoft/orca-math-word-problems-200k
@datasources("microsoft/orca-math-word-problems-200k")
class MathWordProblems200k(BaseAlpacaDataSource):
    dataset_path = "microsoft/orca-math-word-problems-200k"
    instruction_key = "question"
    output_key = "answer"

# MaziyarPanahi/WizardLM_evol_instruct_V2_196k chat
@datasources("cognitivecomputations/WizardLM_evol_instruct_V2_196k_unfiltered_merged_split")
class WizardLM(VicunaChatDataSource):
    def __init__(self) -> None:
        super().__init__("cognitivecomputations/WizardLM_evol_instruct_V2_196k_unfiltered_merged_split")
