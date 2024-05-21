wandb offline
model="HuggingFaceM4/tiny-random-LlamaForCausalLM"
# model="TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"

python -m tuna.launcher.train_flax \
    --mesh fsdp \
    --do_train \
    --task chat-lm \
    --padding max_length \
    --project "test" \
    --run_name "flax-test-lora-1.1b" \
    --dataset="GAIR/lima" \
    --use_lora \
    --packing False \
    --max_length=512 \
    --truncation \
    --truncation_side left \
    --model_name_or_path $model \
    --logging_steps 1 \
    --total_epochs 10 \
    --learning_rate 1e-4 \
    --last_learning_rate_ratio 0.1 \
    --lr_warmup_ratio 0.1 \
    --train_total_batch_size 32 \
    --train_batch_size_per_device 8 \
    --eval_batch_size_per_device 8 \
    --save_strategy no \
    --save_epochs 5 \
    --push_to_hub \
    --output_dir ""