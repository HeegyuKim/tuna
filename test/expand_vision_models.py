from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


# model_name = "Bllossom/llama-3.2-Korean-Bllossom-3B"

def expand_and_push(model_name, codebook_size):
    
    # 토크나이저 로드
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    output_path = "heegyu/" + model_name.split("/")[-1] + "-vis" + str(codebook_size // 1024) + "k"
    codebook_size += 2

    # 새로운 토큰 추가
    new_tokens = ["<begin_of_image>", "<end_of_image>"] + [f"<image_{i}>" for i in range(codebook_size - 2)]
    num_added_tokens = tokenizer.add_tokens(new_tokens)
    print(f"추가된 토큰 수: {num_added_tokens}")

    test_sentence = "이 문장을 이미지로 변환해주세요. <begin_of_image><image_0><image_1><image_2><end_of_image>"
    print(tokenizer.tokenize(test_sentence))
    print(tokenizer.encode(test_sentence))

    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
    model.resize_token_embeddings(len(tokenizer))
    # input_embeddings = model.get_input_embeddings().weight.data
    # average_embedding = torch.mean(input_embeddings[:-num_added_tokens], dim=0)
    # input_embeddings[-num_added_tokens:] = average_embedding[3]


    model.push_to_hub(output_path)
    tokenizer.push_to_hub(output_path)

# expand_and_push("meta-llama/Llama-3.2-1B-Instruct", 65536)
# expand_and_push("meta-llama/Llama-3.2-3B-Instruct", 4096)
expand_and_push("meta-llama/Llama-3.2-8B", 4096)
# expand_and_push("Bllossom/llama-3.2-Korean-Bllossom-3B", 65536)