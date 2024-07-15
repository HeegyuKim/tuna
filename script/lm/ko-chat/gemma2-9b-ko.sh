wandb online

template="gemma"
lr=2e-4
task="chat-lm"
model="google/gemma-2-9b"

train() {
    dataset=$1
    run_name=$2
    
    hub_id="0708-$run_name"

    python -m tuna.launcher.train \
        --mesh sp \
        --do_train \
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
        --total_epochs 2 \
        --learning_rate $lr \
        --lr_warmup_ratio 0.01 \
        --lr_decay_ratio 0.1 \
        --lr_scheduler cosine \
        --train_total_batch_size 32 \
        --train_batch_size_per_device 1 \
        --eval_batch_size_per_device 1 \
        --save_strategy epoch \
        --push_to_hub \
        --push_to_hub_id $hub_id \
        --output_dir ""
}

train "Magpie-Align/Magpie-Pro-MT-300K-v0.1,HAERAE-HUB/qarv-instruct-100k,heegyu/HRC,changpt/ko-lima-vicuna" "0711-gemma2-magpie-qarv-komath"