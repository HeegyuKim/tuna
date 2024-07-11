from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("MLP-KTLim/llama-3-Korean-Bllossom-8B", device_map="cuda", torch_dtype="auto").eval()
tokenizer = AutoTokenizer.from_pretrained("MLP-KTLim/llama-3-Korean-Bllossom-8B")
model.load_adapter("heegyu/ko-prometheus-8b-lora-0708")


PROMETHEUS_PROMPT = """###Task Description:
An instruction (might include an Input inside it), a response to evaluate, a reference answer that gets a score of 5, and a score rubric representing a evaluation criteria are given.
1. Write a detailed feedback that assess the quality of the response strictly based on the given score rubric, not evaluating in general.
2. After writing a feedback, write a score that is an integer between 1 and 5. You should refer to the score rubric.
3. The output format should look as follows: "Feedback: (write a feedback for criteria) [RESULT] (an integer number between 1 and 5)"
4. Please do not generate any other opening, closing, and explanations.

###The instruction to evaluate:
{instruction}

###Response to evaluate:
{response}

###Reference Answer (Score 5):
{reference}

###Score Rubrics:
{rubrics}

###Feedback:"""

RUBRICS = {
    "추론(Reasoning)":"""[언어 모델의 답변이 논리적이고, 일관성 있으며, 깊이 있는 분석을 제공하는가?]
Score 1: 답변이 비논리적이고 일관성이 없으며, 표면적인 분석에 그친다.
Score 2: 답변에 일부 논리적인 요소가 있지만, 일관성이 부족하고 분석이 얕다.
Score 3: 답변이 대체로 논리적이고 일관성이 있으나, 일부 논리적 비약이 있고 분석의 깊이가 보통 수준이다.
Score 4: 답변이 논리적이고 일관성이 있으며, 대체로 깊이 있는 분석을 제공한다. 사소한 논리적 결함이 있을 수 있다.
Score 5: 답변이 완벽하게 논리적이고 일관성이 있으며, 매우 깊이 있고 통찰력 있는 분석을 제공한다.""",
    "수학(Math)": """[언어 모델의 답변이 수학적으로 정확하고, 풀이 과정이 명확하며, 효율적인 해법을 제시하는가?]
Score 1: 답변이 수학적으로 완전히 부정확하고, 풀이 과정이 없거나 이해할 수 없으며, 비효율적인 접근법을 사용한다.
Score 2: 답변에 일부 정확한 수학적 요소가 있지만 중요한 오류가 있고, 풀이 과정이 불완전하며, 비효율적인 해법을 제시한다.
Score 3: 답변이 대체로 수학적으로 정확하지만 일부 오류가 있고, 풀이 과정이 어느 정도 명확하며, 기본적인 효율성을 갖춘 해법을 제시한다.
Score 4: 답변이 수학적으로 정확하고, 풀이 과정이 명확하며, 효율적인 해법을 제시한다. 사소한 개선의 여지가 있을 수 있다.
Score 5: 답변이 수학적으로 완벽하게 정확하고, 풀이 과정이 매우 명확하며, 가장 효율적이고 창의적인 해법을 제시한다.""",
    "글쓰기(Writing)": """[언어 모델의 답변이 구조적이고, 문체가 적절하며, 내용이 풍부하고 창의적인가?]
Score 1: 답변의 구조가 없고, 문체가 부적절하며, 내용이 빈약하고 창의성이 전혀 없다.
Score 2: 답변에 기본적인 구조가 있지만 불완전하고, 문체가 일관성이 없으며, 내용이 제한적이고 창의성이 부족하다.
Score 3: 답변이 어느 정도 구조적이고, 문체가 보통 수준이며, 내용이 적당하고 약간의 창의성이 있다.
Score 4: 답변이 잘 구조화되어 있고, 문체가 적절하며, 내용이 풍부하고 창의적인 요소가 있다.
Score 5: 답변이 완벽하게 구조화되어 있고, 문체가 탁월하며, 내용이 매우 풍부하고 높은 수준의 창의성을 보인다.""",
    "코딩(Coding)": """[언어 모델의 답변이 정확하고, 효율적이며, 가독성 있는 코드를 제공하는가?]
Score 1: 코드가 전혀 작동하지 않고, 비효율적이며, 가독성이 매우 낮다.
Score 2: 코드가 부분적으로 작동하지만 주요 오류가 있고, 효율성이 낮으며, 가독성이 부족하다.
Score 3: 코드가 대체로 작동하고 기본적인 효율성을 갖추었으나, 일부 버그가 있고 가독성이 보통 수준이다.
Score 4: 코드가 정확하게 작동하고 효율적이며, 가독성이 좋다. 사소한 개선의 여지가 있을 수 있다.
Score 5: 코드가 완벽하게 작동하고 매우 효율적이며, 뛰어난 가독성과 최적화된 구조를 갖추고 있다.""",
    "이해(Understanding)": """[언어 모델이 한국어 텍스트의 의미, 뉘앙스, 문맥을 정확하게 이해하고 적절하게 응답하는가?]
Score 1: 한국어 텍스트의 의미를 전혀 이해하지 못하고, 문맥과 뉘앙스를 완전히 놓치며, 부적절한 응답을 한다.
Score 2: 한국어 텍스트의 기본적인 의미만 부분적으로 이해하고, 대부분의 문맥과 뉘앙스를 놓치며, 부적절한 응답이 많다.
Score 3: 한국어 텍스트의 주요 의미를 이해하지만 일부 뉘앙스를 놓치고, 문맥을 부분적으로 파악하며, 대체로 적절한 응답을 한다.
Score 4: 한국어 텍스트의 의미를 정확히 이해하고, 대부분의 뉘앙스와 문맥을 파악하며, 적절한 응답을 한다.
Score 5: 한국어 텍스트의 의미, 뉘앙스, 문맥을 완벽하게 이해하고, 매우 적절하고 세련된 응답을 한다.""",
    "문법(Grammar)":"""[언어 모델의 답변이 한국어 문법 규칙을 정확히 따르고, 적절한 어휘와 표현을 사용하는가?]
Score 1: 심각한 문법 오류가 많고, 부적절한 어휘와 표현을 사용하여 의미 전달이 거의 불가능하다.
Score 2: 중요한 문법 오류가 있고, 제한된 어휘와 부적절한 표현을 사용하여 의미 전달에 어려움이 있다.
Score 3: 일부 문법 오류가 있지만 전반적으로 이해 가능하며, 기본적인 어휘와 표현을 사용한다.
Score 4: 사소한 문법 오류만 있고, 적절한 어휘와 표현을 사용하여 의미를 명확히 전달한다.
Score 5: 완벽한 문법을 구사하고, 다양하고 정확한 어휘와 세련된 표현을 사용하여 의미를 탁월하게 전달한다.""",
}

def judge(instruction, response, reference, rubrics):
    prompt = PROMETHEUS_PROMPT.format(
        instruction=instruction,
        response=response,
        reference=reference,
        rubrics=rubrics
    )
    input_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        return_tensors="pt",
        add_generation_prompt=True
    ).to(model.device)
    judgement = model.generate(
        input_ids, 
        max_new_tokens=256,
        early_stopping=True, 
        eos_token_id=tokenizer.eos_token_id,
        )
    judgement = tokenizer.decode(judgement[0, input_ids.shape[-1]:], skip_special_tokens=True)
    score = int(judgement.split("[RESULT]")[1])
    return judgement, score

print(judge(
    "f(x) = 3x^3 + 2x^2 + 58을 미분하시오.",
    "함수 f(x) = 3x^3 + 2x^2 + 58를 미분하면,\n\nf'(x) = d(3x^3)/dx + d(2x^2)/dx + d(58)/dx\n\n= 3d(x^3)/dx + 2d(x^2)/dx + 0 (상수항은 미분하면 0)\n\n= 3(3x^2) + 2(2x)\n\n= 9x^2 + 4x\n\n따라서, 함수 f(x) = 3x^3 + 2x^2 + 58의 미분값은 f'(x) = 9x^2 + 4x입니다.",
    "9x^2 + 4x",
    RUBRICS["수학(Math)"]
))