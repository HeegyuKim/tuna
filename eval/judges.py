
from llm_blender.pair_ranker.pairrm import DebertaV2PairRM
from transformers import AutoTokenizer
import torch


class BaseJudge:

    def __init__(self, *args, **kwargs):
        pass

    def judge(self, conversation: list[dict], reference: str = None) -> dict:
        raise NotImplementedError("Method judge not implemented")


PROMETHEUS_FORMAT = """###Task Description:
An instruction (might include an Input inside it), a response to evaluate, a reference answer that gets a score of 5, and a score rubric representing a evaluation criteria are given.
1. Write a detailed feedback that assess the quality of the response strictly based on the given score rubric, not evaluating in general.
2. After writing a feedback, write a score that is an integer between 1 and 5. You should refer to the score rubric.
3. The output format should look as follows: \"Feedback: (write a feedback for criteria) [RESULT] (an integer number between 1 and 5)\"
4. Please do not generate any other opening, closing, and explanations.

###The instruction to evaluate: 
{context}

###Response to evaluate: 
{response}

###Reference Answer (Score 5): 
{reference}

###Score Rubrics: 
[Is the model able to provide accurate information, respond promptly, understand user requests, offer quality assistance, and ensure user satisfaction?] 
Score 1: The model frequently provides incorrect or irrelevant information, rarely understands user requests, the assistance provided is generally unhelpful, and the user is very dissatisfied.
Score 2: The model rarely provides correct or relevant information, often fails to understand user requests, the assistance provided is often unhelpful, and the user is dissatisfied.
Score 3: The model sometimes provides correct or relevant information, understands user requests about half the time, the assistance provided is moderately helpful and meets basic user needs, and the user is neither satisfied nor dissatisfied.
Score 4: The model mostly provides accurate information, often understands user requests, the assistance provided is very helpful and usually meets user needs, and the user is satisfied.
Score 5: The model consistently provides accurate and highly relevant information, consistently understands even complex user requests, the assistance provided is exceptionally helpful, exceeding user expectations, and the user is very satisfied."""

class PrometheusJudge:
    
    def __init__(self, model):
        self.model = model

    def format_prompts(self, conversation: list[dict]) -> tuple[str, str]:
        if len(conversation) > 2:
            prompts = []
            for turn in conversation[:-1]:
                prompts.append(turn["role"] + ": " + turn["content"])
            return "\n".join(prompts), conversation[1]['content']
        else:
            return conversation[0]['content'], conversation[1]['content']

    def judge(self, instruction, response, reference: str = None) -> dict:
        if reference is None:
            raise ValueError("Reference answer is required for PrometheusJudge")
        prompt = PROMETHEUS_FORMAT.format(context=instruction, response=response, reference=reference)
        critic = self.model.generate(prompt, greedy=True)
        if "[RESULT]" not in critic:
            return {
                "critic": critic,
                "score": -1
            }
        
        else:
            try:
                score = int(critic.split("[RESULT]")[1])
            except ValueError:
                score = -1
            return {
                "critic": critic,
                "score": score
            }
    
    def judge_conversation(self, conversation: list[dict], reference: str = None) -> dict:
        if reference is None:
            raise ValueError("Reference answer is required for PrometheusJudge")
        context, response = self.format_prompts(conversation)
        return self.judge(context, response, reference)
        


class PairRMJudge(BaseJudge):
    def __init__(self, device: str = "cuda:0", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pairrm = DebertaV2PairRM.from_pretrained("mightbe/Better-PairRM", device_map="cuda:0").eval().half()
        self.tokenizer = AutoTokenizer.from_pretrained("mightbe/Better-PairRM")


    def tokenize_pair(self, instruction, response, reference, source_max_length=2030, candidate_max_length=670):
        ids = []
        max_length = source_max_length + 2 * candidate_max_length
        
        source_prefix = "<|source|>"
        cand1_prefix = "<|candidate1|>"
        cand2_prefix = "<|candidate2|>"

        source_ids = self.tokenizer.encode(source_prefix + instruction, max_length=source_max_length, truncation=True)
        candidate_max_length = (max_length - len(source_ids)) // 2
        candidate1_ids = self.tokenizer.encode(cand1_prefix + response, max_length=candidate_max_length, truncation=True)
        candidate2_ids = self.tokenizer.encode(cand2_prefix + reference, max_length=candidate_max_length, truncation=True)
        
        ids.append(source_ids + candidate1_ids + candidate2_ids)

        encodings = self.tokenizer.pad({"input_ids": ids}, return_tensors="pt", padding="max_length", max_length=max_length)
        return encodings
    
    @torch.no_grad()
    def judge(self, instruction, response, reference: str = None) -> dict:
        if reference is None:
            raise ValueError("Reference answer is required for PairRMJudge")
        
        encodings = self.tokenize_pair(instruction, response, reference)
        encodings = {k:v.to(self.pairrm.device) for k,v in encodings.items()}
        outputs = self.pairrm(**encodings)
        return outputs.logits.tolist()[0]
        # comparison_results = outputs.logits > 0
        # print(logits)
        # print(comparison_results)

        
    def judge_conversation(self, conversation: list[dict], reference: str = None) -> dict:
        if len(conversation) > 2:
            raise ValueError("PairRMJudge only supports a single-turn conversation")
        
        context, response = conversation[-2]['content'], conversation[-1]['content']
        return self.judge(context, response, reference)
        