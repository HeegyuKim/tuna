wandb offline
model="heegyu/TinyLlama__TinyLlama-1.1B-intermediate-step-1431k-3T-tinyllama-1.1b-sft@steps-155897"



train() {
    lr=$1
    beta=$2
    task=$3

    run_name="0510-$task-1.1b"
    
    python -m tuna.launcher.train_flax \
        --mesh sp \
        --do_train \
        --task $task \
        --trainer dpo \
        --padding max_length \
        --project "DDFO-DCO" \
        --dpo_beta $beta \
        --run_name "$run_name-$lr-b$beta" \
        --dataset="dco:heegyu/Ultrafeedback-split-dpo-max-margin" \
        --packing False \
        --truncation \
        --max_length=2048 \
        --use_lora \
        --limit 512 \
        --model_name_or_path $model \
        --total_epochs 1 \
        --learning_rate $lr \
        --last_learning_rate_ratio 0.1 \
        --lr_warmup_ratio 0.1 \
        --train_template zephyr \
        --train_total_batch_size 32 \
        --train_batch_size_per_device 1 \
        --eval_batch_size_per_device 1 \
        --save_strategy no \
        --push_to_hub \
        --push_to_hub_id $run_name \
        --revision_prefix "lr$lr-beta$beta-" \
        --output_dir ""
}

train 2e-4 0.01 dco-v2
# train 2e-4 0.01 dco-v2