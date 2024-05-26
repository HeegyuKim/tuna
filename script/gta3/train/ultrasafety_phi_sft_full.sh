wandb online

model="microsoft/Phi-3-mini-4k-instruct"
template="phi-3"

train() {
    dataset="$1:droussis/UltraSafety_binarized-orpo-dpo,$1:PKU-Alignment/PKU-SafeRLHF-30K"
    task=$2

    hub_id="0521-Phi-3-mini-4k-instruct-$1-full"

    python -m tuna.launcher.train \
        --mesh fsdp \
        --do_train \
        --task $task \
        --padding max_length \
        --model_arch causal-lm \
        --project "GTA3-SAFE-$task" \
        --train_template $template \
        --run_name "$hub_id" \
        --dataset="$dataset" \
        --packing False \
        --trust_remote_code \
        --amp True \
        --max_length=1024 \
        --truncation \
        --model_name_or_path $model \
        --total_epochs 3 \
        --learning_rate 5e-5 \
        --train_total_batch_size 32 \
        --train_batch_size_per_device 1 \
        --eval_batch_size_per_device 1 \
        --save_per_epoch 2 \
        --save_strategy epoch \
        --push_to_hub \
        --push_to_hub_id $hub_id \
        --revision_prefix "lr5e-5" \
        --output_dir ""
}

train "sft" "chat-lm"
