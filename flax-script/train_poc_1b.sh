wandb offline

# model="google/gemma-2-9b"
model="beomi/Llama-3-Open-Ko-8B"
template="llama3"
# model="HuggingFaceM4/tiny-random-LlamaForCausalLM"

python -m tuna.launcher.train_flax \
    --mesh sp \
    --do_train \
    --task chat-lm \
    --padding max_length \
    --project "test" \
    --run_name "llama3-lima" \
    --dataset="GAIR/lima" \
    --train_template $template \
    --packing \
    --packing_strategy pad \
    --max_length=2048 \
    --truncation \
    --model_name_or_path $model \
    --total_epochs 3 \
    --lr_warmup_ratio 0.05 \
    --last_learning_rate 0.1 \
    --learning_rate 5e-5 \
    --train_total_batch_size 32 \
    --train_batch_size_per_device 1 \
    --eval_batch_size_per_device 1 \
    --save_strategy no \
    --push_to_hub \
    --output_dir ""