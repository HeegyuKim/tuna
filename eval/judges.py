
from transformers import AutoTokenizer
import torch
from .prometheus_prompts import get_prompt_template, load_rubric, ABS_SYSTEM_PROMPT


class BaseJudge:

    def __init__(self, *args, **kwargs):
        pass

    def judge(self, conversation: list[dict], reference: str = None) -> dict:
        raise NotImplementedError("Method judge not implemented")


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

    def judge(self, instruction, response, reference: str = None, rubric: str = "helpfulness") -> dict:
        include_reference = True if reference else False
        rubric = load_rubric(rubric, "absolute")
        prompt = get_prompt_template("absolute", include_reference).format(instruction=instruction, response=response, reference_answer=reference, rubric=rubric)

        critic = self.model.generate(ABS_SYSTEM_PROMPT + "\n\n" + prompt, greedy=True)

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
    
    def judge_conversation(self, conversation: list[dict], reference: str = None, rubric: str = "helpfulness") -> dict:
        context, response = self.format_prompts(conversation)
        return self.judge(context, response, reference, rubric)
        


class PairRMJudge(BaseJudge):
    def __init__(self, device: str = "cuda:0", *args, **kwargs):
        super().__init__(*args, **kwargs)
        from llm_blender.pair_ranker.pairrm import DebertaV2PairRM
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
        