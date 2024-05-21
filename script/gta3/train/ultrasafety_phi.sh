wandb online

model="microsoft/Phi-3-mini-4k-instruct"
template="phi-3"

train() {
    dataset="$1:droussis/UltraSafety_binarized-orpo-dpo,$1:PKU-Alignment/PKU-SafeRLHF-30K"
    task=$2
    use_lora=$3
    if $use_lora; then
        run_name="0521-Phi-3-mini-4k-instruct-safety-$task-lora-5e-5"
    else
        run_name="0521-Phi-3-mini-4k-instruct-safety-$task-5e-5"
    fi

    python -m tuna.launcher.train \
        --mesh sp \
        --do_train \
        --task $task \
        --padding max_length \
        --model_arch causal-lm \
        --project "GTA3-GATE" \
        --train_template $template \
        --run_name "$run_name" \
        --dataset="$dataset" \
        --packing True \
        --trust_remote_code \
        --amp \
        --use_lora $use_lora \
        --max_length=1024 \
        --truncation \
        --model_name_or_path $model \
        --total_epochs 2 \
        --learning_rate 5e-5 \
        --train_total_batch_size 32 \
        --train_batch_size_per_device 1 \
        --eval_batch_size_per_device 1 \
        --save_epochs 2 \
        --save_strategy epoch \
        --push_to_hub \
        --output_dir ""
}

train "sft" "chat-lm" true
train "sft" "chat-lm" false
