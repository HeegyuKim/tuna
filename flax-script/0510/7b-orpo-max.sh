wandb online
model="mistralai/Mistral-7B-v0.1"
run_name="0510-orpo-7b"

train() {
    lr=$1
    beta=$2

    python -m tuna.launcher.train_flax \
        --mesh fsdp \
        --do_train \
        --task orpo \
        --padding max_length \
        --project "DDFO-ORPO" \
        --run_name "$run_name-lr$lr-beta$beta" \
        --dataset="dpo:heegyu/UltraFeedback-max-margin" \
        --packing False \
        --truncation \
        --max_length=2048 \
        --model_name_or_path $model \
        --total_epochs 3 \
        --learning_rate $lr \
        --orpo_beta $beta \
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


for lr in 5e-6 1e-5 2e-5 5e-5; do
    train $lr 0.1
done