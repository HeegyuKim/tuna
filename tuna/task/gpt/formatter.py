
class BaseFormatter:
    def handle_uttr(self, uttr, tokenizer):
        return uttr

    def join_convs(self, convs, tokenizer):
        return "".join(convs)
    
    def format_uttrs(self, convs, tokenizer, for_generation=False):
        out_convs = []
        for c in convs:
            c = self.handle_uttr(c, tokenizer)
            if isinstance(c, list):
                out_convs.extend(c)
            else:
                out_convs.append(c)
        
        if for_generation:
            return out_convs[:-1]
        else:
            return out_convs
    
    def format(self, convs, tokenizer, for_generation=False):
        out_convs = self.format_uttrs(convs, tokenizer, for_generation)
        return self.join_convs([x["value"] for x in out_convs], tokenizer)
    

class SimpleFormatter(BaseFormatter):
    system_format="System: {value} {eos}\n\n"
    human_format="Human: {value} {eos}\n\n"
    assistant_prefix="Assistant:"
    assistant_format="{value} {eos}\n\n"

    def handle_uttr(self, uttr, tokenizer):
        eos = tokenizer.eos_token
        who, text = uttr.get("role") or uttr["from"], uttr.get("content") or uttr["value"]

        if who in ["human", "user"]:
            return {
                "from": "human",
                "value": self.human_format.format(value=text, eos=eos)
            }
        elif who == "system":
            return {
                "from": "system",
                "value": self.system_format.format(value=text, eos=eos)
            }
        else:
            return [
                {
                    "from": "human",
                    "value": self.assistant_prefix
                },
                {
                    "from": "gpt",
                    "value": self.assistant_format.format(value=text, eos=eos)
                }
                ]

# <|system|>
# You are a friendly chatbot who always responds in the style of a pirate.</s>
# <|user|>
# How many helicopters can a human eat in one sitting?</s>
# <|assistant|>
# Ah, me hearty matey! But yer question be a puzzler! A human cannot eat a helicopter in one sitting, as helicopters are not edible. They be made of metal, plastic, and other materials, not food!
class ZephyrFormatter(SimpleFormatter):
    system_format="<|system|>\n{value}{eos}\n"
    human_format="<|user|>\n{value}{eos}\n"
    assistant_prefix="<|assistant|>\n"
    assistant_format="{value}{eos}\n"

class LlamaFormatter(SimpleFormatter):
    system_format="<<SYS>>\n{value}\n<</SYS>>\n\n"
    human_format="[INST] {value} [/INST]"
    assistant_prefix=""
    assistant_format="{value}{eos}"

class VicunaFormatter(SimpleFormatter):
    system_format="{value}\n\n"
    human_format="User: {value}\n"
    assistant_prefix="Assistant:"
    assistant_format="{value}{eos}"

class Openchat3_5Formatter(SimpleFormatter):
    system_format="{value}\n\n"
    human_format="GPT4 Correct User: {value}\n"
    assistant_prefix="GPT4 Correct Assistant:"
    assistant_format="{value}{eos}"

class HD42DotFormatter(SimpleFormatter):
    system_format="{value}\n\n"
    human_format="<human>:\n{value}\n\n"
    assistant_prefix="<bot>:\n"
    assistant_format="{value}{eos}\n\n"

class En2KoMTFormatter(SimpleFormatter):
    system_format="{value}\n\n"
    human_format="### English:\n{value}\n\n"
    assistant_prefix="### Korean:\n"
    assistant_format="{value}{eos}\n\n"

class ChatMLFormatter(SimpleFormatter):
    system_format="<|im_start|>system\n{value}<|im_end|>\n"
    human_format="<|im_start|>user\n{value}<|im_end|>\n"
    assistant_prefix="<|im_start|>assistant\n"
    assistant_format="{value}{eos}\n"

class BertRewardModelFormatter(SimpleFormatter):
    system_format=""
    human_format="[CLS] {value}"
    assistant_prefix=" [SEP] "
    assistant_format="{value} [SEP]"




FORMATTERS = {
    "simple": SimpleFormatter,
    "zephyr": ZephyrFormatter,
    "llama": LlamaFormatter,
    "42dot": HD42DotFormatter,
    "chatml": ChatMLFormatter,
    "en2ko-mt": En2KoMTFormatter,
    "bert-rm": BertRewardModelFormatter,
    "openchat-3.5": Openchat3_5Formatter,
    "vicuna": VicunaFormatter,
}

VICUNA_FORMAT = """### Human:
{human}

### Assistant:
{bot}"""     

LDCC_FORMAT = """### Prompt:
{human}

### Answer:
{bot}"""     

HD42DOT_FORMAT = """<human>: 
{human}
<bot>: 
{bot}"""     

NO_FORMAT = "{human}{bot}"

CONTEXT_FORMATS = {
    "vicuna": VICUNA_FORMAT,
    "42dot": HD42DOT_FORMAT,
    "ldcc": LDCC_FORMAT,
}


def build_prompt(conversations, response, prompt_message = None, eos_token="", format_id="vicuna"):
    context_format = CONTEXT_FORMATS[format_id]
    context = []

    for conv in conversations:
        context.append(context_format.format(human=conv["human"], bot=conv["gpt"]) + eos_token)
    
    prompt = "\n\n".join(context) + response
    if prompt_message:
        prompt = f"{prompt_message}\n\n{context}"
    return prompt