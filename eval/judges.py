

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
        score = int(critic.split("[RESULT]")[1])
        return {
            "critic": critic,
            "score": score
        }
    
    def judge_conversation(self, conversation: list[dict], reference: str = None) -> dict:
        if reference is None:
            raise ValueError("Reference answer is required for PrometheusJudge")
        context, response = self.format_prompts(conversation)
        return self.judge(context, response, reference)
        
