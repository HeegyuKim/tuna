TEMPLATE_42DOT = "{% for message in messages %}{% if message['role'] == 'user' %}{{ '<human>:\n' }}{% elif message['role'] == 'assistant' %}{{ '<bot>:\n' }}{% endif %}{{ message['content'] + '\n' }}{% endfor %}{% if add_generation_prompt %}{{ '<bot>:\n' }}{% endif %}"
TEMPLATE_VICUNA = "{% for message in messages %}{% if message['role'] == 'user' %}{{ '### User:\n' }}{% elif message['role'] == 'assistant' %}{{ '### Assistant:\n' }}{% endif %}{{ message['content'] + '\n\n' }}{% endfor %}{% if add_generation_prompt %}{{ '### Assistant:\n' }}{% endif %}"
TEMPLATE_NEW_VICUNA = "{% if messages[0]['role'] == 'system' %}{% set loop_messages = messages[1:] %}{% set system_message = messages[0]['content'] %}{% else %}{% set loop_messages = messages %}{% set system_message = 'A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user\\'s questions.' %}{% endif %}{% for message in loop_messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if loop.index0 == 0 %}{{ system_message }}{% endif %}{% if message['role'] == 'user' %}{{ ' USER: ' + message['content'].strip() }}{% elif message['role'] == 'assistant' %}{{ ' ASSISTANT: ' + message['content'].strip() + eos_token }}{% endif %}{{ '\n' }}{% endfor %}{% if add_generation_prompt %}{{ ' ASSISTANT:' }}{% endif %}"

TEMPLATE_LLAMA_3 = "{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% endif %}"

# TEMPLATE_LLAMA = "{% if messages[0]['role'] == 'system' %}{% set loop_messages = messages[1:] %}{% set system_message = messages[0]['content'] %}{% else %}{% set loop_messages = messages %}{% set system_message = false %}{% endif %}{% for message in loop_messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if loop.index0 == 0 and system_message != false %}{% set content = '<<SYS>>\\n' + system_message + '\\n<</SYS>>\\n\\n' + message['content'] %}{% else %}{% set content = message['content'] %}{% endif %}{% if message['role'] == 'user' %}{{ bos_token + '[INST] ' + content.strip() + ' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ ' '  + content.strip() + ' ' + eos_token }}{% endif %}{% endfor %}"
TEMPLATE_LLAMA = (
            "{% if messages[0]['role'] == 'system' %}"
            "{% set loop_messages = messages[1:] %}"  # Extract system message if it's present
            "{% set system_message = messages[0]['content'] %}"
            "{% elif USE_DEFAULT_PROMPT == true and not '<<SYS>>' in messages[0]['content'] %}"
            "{% set loop_messages = messages %}"  # Or use the default system message if the flag is set
            "{% set system_message = 'DEFAULT_SYSTEM_MESSAGE' %}"
            "{% else %}"
            "{% set loop_messages = messages %}"
            "{% set system_message = false %}"
            "{% endif %}"
            "{% if loop_messages|length == 0 and system_message %}"  # Special handling when only sys message present
            "{{ bos_token + '[INST] <<SYS>>\\n' + system_message + '\\n<</SYS>>\\n\\n [/INST]' }}"
            "{% endif %}"
            "{% for message in loop_messages %}"  # Loop over all non-system messages
            "{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}"
            "{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}"
            "{% endif %}"
            "{% if loop.index0 == 0 and system_message != false %}"  # Embed system message in first message
            "{% set content = '<<SYS>>\\n' + system_message + '\\n<</SYS>>\\n\\n' + message['content'] %}"
            "{% else %}"
            "{% set content = message['content'] %}"
            "{% endif %}"
            "{% if message['role'] == 'user' %}"  # After all of that, handle messages/roles in a fairly normal way
            "{{ bos_token + '[INST] ' + content.strip() + ' [/INST]' }}"
            "{% elif message['role'] == 'system' %}"
            "{{ '<<SYS>>\\n' + content.strip() + '\\n<</SYS>>\\n\\n' }}"
            "{% elif message['role'] == 'assistant' %}"
            "{{ ' '  + content.strip() + ' ' + eos_token }}"
            "{% endif %}"
            "{% endfor %}"
        )
TEMPLATE_ZEPHYR = "{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '<|user|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'system' %}\n{{ '<|system|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'assistant' %}\n{{ '<|assistant|>\n'  + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}"
TEMPLATE_CHATML = "{% for message in messages %}{% if message['role'] == 'user' %}{{ '<|im_start|>user\n' }}{% elif message['role'] == 'assistant' %}{{ '<|im_start|>assistant\n' }}{% endif %}{{ message['content'] + '<|im_end|>\n' }}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant:\n' }}{% endif %}"

PROMPT_TEMPLATES = {
    "chatml": TEMPLATE_CHATML,
    
    "42dot": TEMPLATE_42DOT,
    "42dot/42dot_LLM-SFT-1.3B": TEMPLATE_42DOT,
    
    "vicuna": TEMPLATE_VICUNA,
    "solar": TEMPLATE_VICUNA,
    "lmsys/vicuna-7b-v1.5": TEMPLATE_NEW_VICUNA,
    "declare-lab/starling-7B": TEMPLATE_NEW_VICUNA,
    "Locutusque/TinyMistral-248M-v2-Instruct": TEMPLATE_CHATML,
    
    "llama": TEMPLATE_LLAMA,
    "meta-llama/Llama-2-7b-chat-hf": TEMPLATE_LLAMA,
    
    "zephyr": TEMPLATE_ZEPHYR,
}