

python -m tuna.serve.gradio_flax \
    --model_name microsoft/Phi-3-mini-4k-instruct \
    --fully_sharded_data_parallel False \
    --mesh sp --eos_token "<|end|>"

python -m tuna.serve.gradio_flax \
    --model_name heegyu/TinyMistral-248M-v2.5-Instruct-orpo@epoch-1 \
    --fully_sharded_data_parallel False \
    --chat_template chatml 
    
    \
    --fully_sharded_data_parallel False \
    --mesh sp
    