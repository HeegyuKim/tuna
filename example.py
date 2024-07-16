# Use a pipeline as a high-level helper
from transformers import pipeline

messages = [
    {"role": "user", "content": "Who are you?"},
]
pipe = pipeline("text-generation", model="heegyu/gemma-2-9b-lima", device_map="auto", torch_dtype="auto")
print(pipe(messages, max_new_tokens=128, eos_token_id=107))