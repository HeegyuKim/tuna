wandb offline
model="heegyu/0710-qwen2-magpie-qarv-komath"
run_name="0710-qwen2-magpie-qarv-komath-dpo"

train() {
    lr=$1
    beta=$2
    
    python -m tuna.launcher.train_flax \
        --mesh sp \
        --do_train \
        --task dpo \
        --trainer dpo \
        --padding max_length \
        --project "KoChat-DPO" \
        --run_name "$run_name-lr$lr-beta$beta" \
        --dataset="dpo:Magpie-Align/Magpie-Pro-DPO-200K" \
        --packing False \
        --truncation \
        --max_length=2048 \
        --dpo_prompt_length=1024 \
        --dpo_response_length=1024 \
        --model_name_or_path $model \
        --total_epochs 1 \
        --learning_rate $lr \
        --dpo_beta $beta \
        --last_learning_rate_ratio 0.1 \
        --lr_warmup_ratio 0.03 \
        --train_template chatml \
        --train_total_batch_size 32 \
        --train_batch_size_per_device 1 \
        --eval_batch_size_per_device 1 \
        --save_strategy epoch \
        --push_to_hub \
        --push_to_hub_id $run_name \
        --revision_prefix "lr$lr-beta$beta-" \
        --output_dir ""
}


for lr in 5e-6; do
    train $lr 0.25
    train $lr 0.1
    train $lr 0.5
    train $lr 1.0
done