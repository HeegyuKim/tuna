wandb online
model="google/gemma-2b"
run_name="0517-orpo-2b"

train() {
    lr=$1
    beta=$2

    python -m tuna.launcher.train_flax \
        --mesh sp \
        --do_train \
        --task orpo \
        --padding max_length \
        --project "DDFO-ORPO" \
        --run_name "$run_name-lr$lr-beta$beta" \
        --dataset="dpo:ultrafeedback,dpo:openbmb/UltraInteract_pair" \
        --packing False \
        --truncation \
        --max_length=2048 \
        --orpo_prompt_length=1024 \
        --orpo_response_length=1024 \
        --model_name_or_path $model \
        --total_epochs 3 \
        --learning_rate $lr \
        --orpo_beta $beta \
        --last_learning_rate_ratio 0.1 \
        --lr_warmup_ratio 0.1 \
        --train_template gemma \
        --filter_no_bos \
        --train_total_batch_size 32 \
        --train_batch_size_per_device 2 \
        --eval_batch_size_per_device 2 \
        --save_strategy epoch \
        --push_to_hub \
        --push_to_hub_id $run_name \
        --revision_prefix "lr$lr-beta$beta-" \
        --output_dir ""
}

# train 2e-5 0.1
for lr in 2e-5 5e-5; do
    train $lr 0.1
done