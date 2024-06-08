wandb online

model="microsoft/Phi-3-mini-4k-instruct"
template="phi-3"

train() {
    dataset="$1:droussis/UltraSafety_binarized-orpo-dpo"
    task=$2
    beta=$3
    run_name="0528-Phi-3-mini-4k-instruct-safety-$task-full-flax"

    python -m tuna.launcher.train_flax \
        --mesh sp \
        --task $task \
        --do_train \
        --padding max_length \
        --project "GTA3-SAFE-$2" \
        --train_template $template \
        --run_name "$run_name-beta$beta" \
        --dataset="$dataset" \
        --packing false \
        --max_length=2048 \
        --orpo_prompt_length 1024 \
        --orpo_response_length 1024 \
        --truncation \
        --model_name_or_path $model \
        --total_epochs 3 \
        --learning_rate 2e-5 \
        --last_learning_rate_ratio 0.1 \
        --lr_warmup_ratio 0.1 \
        --train_total_batch_size 32 \
        --train_batch_size_per_device 1 \
        --eval_batch_size_per_device 1 \
        --orpo_beta $beta \
        --filter_no_bos \
        --save_strategy epoch \
        --revision_prefix "lr2e-5-beta$beta" \
        --push_to_hub \
        --output_dir ""

}

train "dpo" "orpo" 0.25
train "dpo" "orpo" 0.5
# train "dpo" "orpo" 0.1
# train "dpo" "orpo" 0
