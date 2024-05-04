wandb online
model="HuggingFaceH4/mistral-7b-sft-beta"

    # --gradient_checkpointing everything_saveable \

train() {
    dataset=$1
    lr=$2
    
    python -m tuna.launcher.train_flax \
        --mesh sp \
        --do_train \
        --task dpo \
        --trainer dpo \
        --padding max_length \
        --project "DDFO-DPO" \
        --run_name "0504-mistral-7b-sft-beta-$dataset-$lr" \
        --dataset="dpo:heegyu/UltraFeedback-$dataset" \
        --packing False \
        --truncation \
        --max_length=2048 \
        --use_lora \
        --model_name_or_path $model \
        --total_epochs 3 \
        --learning_rate $lr \
        --last_learning_rate_ratio 0.1 \
        --lr_warmup_ratio 0.1 \
        --train_template zephyr \
        --train_total_batch_size 32 \
        --train_batch_size_per_device 1 \
        --eval_batch_size_per_device 1 \
        --save_strategy epoch \
        --push_to_hub \
        --output_dir ""
}

train "max-margin" 5e-5
train "max-margin" 1e-5
train "feedback-tree-3" 5e-5
train "feedback-tree-3" 1e-5