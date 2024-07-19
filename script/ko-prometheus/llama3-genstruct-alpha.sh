wandb online

lr=2e-4
task="chat-lm"

# template="llama3"
# model="MLP-KTLim/llama-3-Korean-Bllossom-8B"

template="chatml"
model="Qwen/Qwen2-7B-Instruct"

train() {
    dataset=$1
    run_name=$2
    
    hub_id="0708-$run_name"

    python -m tuna.launcher.train \
        --mesh sp \
        --do_train --do_eval \
        --task $task \
        --padding max_length \
        --model_arch causal-lm \
        --project "KoChat-SFT" \
        --train_template $template \
        --run_name "$run_name" \
        --dataset="$dataset" \
        --packing False \
        --trust_remote_code \
        --amp True \
        --use_lora \
        --max_length=2048 \
        --truncation \
        --model_name_or_path $model \
        --total_epochs 3 \
        --learning_rate $lr \
        --lr_warmup_ratio 0.01 \
        --lr_decay_ratio 0.1 \
        --train_total_batch_size 32 \
        --train_batch_size_per_device 1 \
        --eval_batch_size_per_device 1 \
        --save_strategy epoch \
        --push_to_hub \
        --push_to_hub_id $hub_id \
        --output_dir ""
}

# train "heegyu/K2-Feedback-splited" "ko-prometheus"
train heegyu/K2-Feedback-splited "ko-prometheus-7b"
# train "beomi/KoAlpaca-v1.1a" "ko-alpaca"