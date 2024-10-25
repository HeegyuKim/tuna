# reasoning heegyu/wildguardmix-train-reasoning-cleaned-125k-1005-messages
# direct heegyu/wildguardmix-direct-1001
wandb online


model="meta-llama/Llama-3.2-1B-Instruct"

train() {
    lr=$1
    dataset=$2
    run_name=$3
    python -m tuna.launcher.train \
        --do_train \
        --task chat-lm \
        --padding max_length \
        --model_arch causal-lm \
        --project "llama-barrier" \
        --train_template llama3 \
        --run_name "$run_name-lr$lr" \
        --dataset="$dataset" \
        --packing False \
        --amp \
        --logging_steps 256 \
        --max_length=2048 \
        --truncation \
        --model_name_or_path $model \
        --total_epochs 3 \
        --learning_rate $lr \
        --train_total_batch_size 128 \
        --train_batch_size_per_device 2 \
        --eval_batch_size_per_device 2 \
        --save_strategy epoch \
        --push_to_hub False \
        --revision_prefix "lr$lr-bs128" \
        --output_dir "/data/checkpoint/"
}

train 1e-5 "hf-chat:heegyu/wildguardmix-direct-1001" "1025-barrier-1B-direct"
train 5e-6 "hf-chat:heegyu/wildguardmix-direct-1001" "1025-barrier-1B-direct"
