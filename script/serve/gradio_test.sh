

python -m tuna.serve.gradio_flax \
    --model_name Felladrin/TinyMistral-248M-Chat-v2 \
    --fully_sharded_data_parallel False \
    --mesh sp

python -m tuna.serve.gradio_flax \
    --model_name heegyu/TinyLlama-1.1b-max-margin@epoch-3 \
    --chat_template zephyr 
    
    \
    --fully_sharded_data_parallel False \
    --mesh sp
    