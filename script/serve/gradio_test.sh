

python -m tuna.serve.gradio_flax \
    --model_name Felladrin/TinyMistral-248M-Chat-v2 \
    --fully_sharded_data_parallel False \
    --mesh sp

python -m tuna.serve.gradio_flax \
    --model_name heegyu/TinyLlama__TinyLlama-1.1B-intermediate-step-1431k-3T-tinyllama-1.1b-sft@steps-155897 \
    --chat_template zephyr 
    
    \
    --fully_sharded_data_parallel False \
    --mesh sp
    