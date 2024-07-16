wandb online
model="HuggingFaceH4/mistral-7b-sft-beta"


train() {
    lr=$1
    datasets=$2
    run_name=$3

    python -m tuna.launcher.train_flax \
        --mesh sp \
        --do_train \
        --task orpo \
        --padding max_length \
        --project "DDFO-ORPO" \
        --run_name "$run_name-lr$lr" \
        --dataset="$datasets" \
        --packing False \
        --truncation \
        --max_length=2048 \
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
        --push_to_hub_id "$run_name" \
        --revision_prefix "lr$lr-beta0.1-" \
        --output_dir ""
}

# train 5e-6 "dpo:heegyu/Ultrafeedback-max-margin-critique:cot" "0625-zephyr7b-orpo-cot"
train 5e-6 "dpo:heegyu/Ultrafeedback-max-margin" "0625-zephyr7b-orpo"