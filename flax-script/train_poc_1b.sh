wandb online
# export HF_HOME=/data-plm/hf-home
# model="TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
model="beomi/Llama-3-Open-Ko-8B"
# model="HuggingFaceM4/tiny-random-LlamaForCausalLM"

python -m tuna.launcher.train_flax \
    --mesh sp \
    --do_train \
    --task chat-lm \
    --padding max_length \
    --project "test" \
    --run_name "Llama3-open-ko-adafactor" \
    --dataset="GAIR/lima" \
    --packing False \
    --max_length=1024 \
    --param_dtype fp32 \
    --truncation \
    --optimizer adafactor \
    --model_name_or_path $model \
    --logging_steps 1 \
    --total_epochs 3 \
    --learning_rate 5e-5 \
    --train_total_batch_size 32 \
    --train_batch_size_per_device 1 \
    --eval_batch_size_per_device 1 \
    --save_strategy no \
    --push_to_hub \
    --output_dir ""