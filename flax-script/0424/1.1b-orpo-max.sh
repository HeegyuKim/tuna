wandb online
model="TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"

python -m tuna.launcher.train_flax \
    --mesh fsdp \
    --do_train \
    --task orpo \
    --padding max_length \
    --project "DDFO-ORPO" \
    --run_name "TinyLlama-1.1b-max-margin-0427-orpo" \
    --dataset="dpo:heegyu/UltraFeedback-max-margin" \
    --packing False \
    --truncation \
    --truncation_side left \
    --max_length=2048 \
    --model_name_or_path $model \
    --total_epochs 3 \
    --learning_rate 5e-5 \
    --last_learning_rate_ratio 0.1 \
    --lr_warmup_ratio 0.1 \
    --train_template zephyr \
    --train_total_batch_size 32 \
    --train_batch_size_per_device 1 \
    --eval_batch_size_per_device 1 \
    --save_strategy epoch \
    --push_to_hub \
    --output_dir ""