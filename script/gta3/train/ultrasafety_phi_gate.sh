wandb online

model="microsoft/Phi-3-mini-4k-instruct"
template="phi-3"

train() {
    dataset="$1:droussis/UltraSafety_binarized-orpo-dpo,$1:PKU-Alignment/PKU-SafeRLHF-30K,llamaguard-cot-rev:iknow-lab/GTA3-promptguard-v0516"
    task=$2
    lora_r=$3

    hub_id="0522-Phi-3-mini-4k-instruct-gate-$task-lora"

    python -m tuna.launcher.train \
        --mesh sp \
        --do_train \
        --task $task \
        --padding max_length \
        --model_arch causal-lm \
        --project "GTA3-GATE" \
        --train_template $template \
        --run_name "$hub_id-r$lora_r" \
        --dataset="$dataset" \
        --packing False \
        --trust_remote_code \
        --amp True \
        --use_lora $use_lora \
        --lora_r $lora_r \
        --lora_alpha $((lora_r * 2)) \
        --max_length=1024 \
        --truncation \
        --model_name_or_path $model \
        --total_epochs 3 \
        --learning_rate 5e-5 \
        --train_total_batch_size 32 \
        --train_batch_size_per_device 4 \
        --eval_batch_size_per_device 4 \
        --save_per_epoch 2 \
        --save_strategy epoch \
        --push_to_hub \
        --push_to_hub_id $hub_id \
        --revision_prefix "r$lora_r-5e-5" \
        --output_dir ""
}

# train "sft" "chat-lm" 8
for r in 8 32 128; do
    train "sft" "chat-lm" $r
done
