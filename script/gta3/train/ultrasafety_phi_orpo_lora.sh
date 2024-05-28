wandb offline

model="microsoft/Phi-3-mini-4k-instruct"
template="phi-3"

train() {
    dataset="$1:droussis/UltraSafety_binarized-orpo-dpo" #,$1:PKU-Alignment/PKU-SafeRLHF-30K"
    task=$2
    lora_r=$3
    beta=$4

    hub_id="0528-Phi-3-mini-4k-instruct-safety-$task-lora"

    python -m tuna.launcher.train \
        --mesh fsdp \
        --do_train \
        --task $task \
        --padding max_length \
        --model_arch causal-lm \
        --project "GTA3-SAFE-$1" \
        --train_template $template \
        --run_name "$hub_id-r$lora_r-beta$beta" \
        --dataset="$dataset" \
        --packing False \
        --trust_remote_code \
        --amp True \
        --use_lora \
        --lora_r $lora_r \
        --lora_alpha $((lora_r * 2)) \
        --max_length=1024 \
        --truncation \
        --model_name_or_path $model \
        --total_epochs 3 \
        --orpo_beta $beta \
        --learning_rate 5e-5 \
        --train_total_batch_size 32 \
        --train_batch_size_per_device 1 \
        --eval_batch_size_per_device 1 \
        --save_per_epoch 2 \
        --save_strategy epoch \
        --push_to_hub \
        --push_to_hub_id $hub_id \
        --revision_prefix "r$lora_r-5e-5-beta$beta" \
        --output_dir ""
}

train "dpo" "orpo" 8 0.1
train "dpo" "orpo" 8 0

# for r in 32 128; do
#     train "dpo" "dpo" $r
# done
