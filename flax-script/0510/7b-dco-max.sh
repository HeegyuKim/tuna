wandb online
model="alignment-handbook/zephyr-7b-sft-full"


    # --gradient_checkpointing everything_saveable \

train() {
    dataset=$1
    lr=$2
    beta=$3
    task=$4

    run_name="0510-$task-7b"
    
    python -m tuna.launcher.train_flax \
        --mesh sp \
        --do_train \
        --task $task \
        --trainer dpo \
        --padding max_length \
        --project "DDFO-DPO" \
        --dpo_beta $beta \
        --run_name "$run_name-$dataset-$lr-b$beta" \
        --dataset="dpo:heegyu/UltraFeedback-$dataset" \
        --packing False \
        --truncation \
        --max_length=2048 \
        --use_lora \
        --model_name_or_path $model \
        --total_epochs 1 \
        --learning_rate $lr \
        --last_learning_rate_ratio 0.1 \
        --lr_warmup_ratio 0.1 \
        --train_template zephyr \
        --train_total_batch_size 32 \
        --train_batch_size_per_device 1 \
        --eval_batch_size_per_device 1 \
        --save_strategy epoch \
        --push_to_hub \
        --push_to_hub_id $run_name \
        --revision_prefix "lr$lr-beta$beta-" \
        --output_dir ""
}

train "max-margin" 2e-4 0.01 dco-v4
train "max-margin" 2e-4 0.01 dco-v2