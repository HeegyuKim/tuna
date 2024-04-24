wandb online
model="HuggingFaceH4/mistral-7b-sft-beta"

python -m tuna.launcher.train_flax \
    --mesh fsdp \
    --do_train \
    --task dpo \
    --trainer dpo \
    --padding max_length \
    --project "DDFO-DPO" \
    --run_name "mistral-7b-sft-beta-feedback-tree-3-epoch3-0424" \
    --dataset="dpo:heegyu/UltraFeedback-feedback-tree-3" \
    --packing False \
    --truncation \
    --truncation_side left \
    --max_length=2048 \
    --model_name_or_path $model \
    --total_epochs 3 \
    --logging_steps 128 \
    --learning_rate 1e-5 \
    --last_learning_rate_ratio 0.1 \
    --lr_warmup_ratio 0.1 \
    --train_template zephyr \
    --train_total_batch_size 32 \
    --train_batch_size_per_device 1 \
    --eval_batch_size_per_device 1 \
    --save_strategy epoch \
    --push_to_hub \
    --output_dir ""