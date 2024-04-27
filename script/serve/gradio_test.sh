

python -m tuna.serve.gradio_flax \
    --model_name heegyu/TinyMistral-248M-v2.5-orpo@epoch-3 \
    --chat_template zephyr \
    --fully_sharded_data_parallel False \
    --mesh sp

python -m tuna.serve.gradio_flax \
    --model_name heegyu/TinyLlama-1.1b-max-margin@epoch-3 \
    --chat_template zephyr 
    
    \
    --fully_sharded_data_parallel False \
    --mesh sp
    