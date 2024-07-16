wandb online
model="microsoft/Phi-3-mini-4k-instruct"
run_name="0601-phi-3b-dpo-feedback-tree"

train() {
    lr=$1
    beta=$2
    
    rm -rf ~/.cache/huggingface/datasets

    python -m tuna.launcher.train_flax \
        --mesh sp \
        --do_train \
        --task dpo \
        --trainer dpo \
        --padding max_length \
        --project "DDFO-DPO" \
        --run_name "$run_name-lr$lr-beta$beta" \
        --dataset="dpo:heegyu/UltraFeedback-feedback-tree-3" \
        --packing False \
        --truncation \
        --max_length=1536 \
        --dpo_prompt_length=768 \
        --dpo_response_length=768 \
        --model_name_or_path $model \
        --total_epochs 1 \
        --learning_rate $lr \
        --dpo_beta $beta \
        --last_learning_rate_ratio 0.1 \
        --lr_warmup_ratio 0.1 \
        --train_template phi-3 \
        --filter_no_bos \
        --train_total_batch_size 32 \
        --train_batch_size_per_device 1 \
        --eval_batch_size_per_device 1 \
        --save_strategy epoch \
        --push_to_hub \
        --push_to_hub_id $run_name \
        --revision_prefix "lr$lr-beta$beta-" \
        --output_dir ""
}

# train 2e-5 0.1
for lr in 5e-6 1e-5; do
    train $lr 0.1
    train $lr 0.25
    train $lr 0.5
done