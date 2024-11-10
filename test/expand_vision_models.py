from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


# 4096 + <begin_of_image> + <end_of_image>
num_titok_tokens = 4096 + 2

codebook_size = num_titok_tokens
model_name = "Bllossom/llama-3.2-Korean-Bllossom-3B"
output_path = "heegyu/llama-3.2-Korean-Bllossom-3B-vision-expanded"

# 토크나이저 로드
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 새로운 토큰 추가
new_tokens = ["<begin_of_image>", "<end_of_image>"] + [f"<image_{i}>" for i in range(codebook_size - 2)]
num_added_tokens = tokenizer.add_tokens(new_tokens)
print(f"추가된 토큰 수: {num_added_tokens}")

test_sentence = "이 문장을 이미지로 변환해주세요. <begin_of_image><image_0><image_1><image_2><end_of_image>"
print(tokenizer.tokenize(test_sentence))
print(tokenizer.encode(test_sentence))

model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
model.resize_token_embeddings(len(tokenizer))
input_embeddings = model.get_input_embeddings().weight.data
average_embedding = torch.mean(input_embeddings[:-num_added_tokens], dim=0)
input_embeddings[-num_added_tokens:] = average_embedding[3]


model.push_to_hub(output_path)
tokenizer.push_to_hub(output_path)
