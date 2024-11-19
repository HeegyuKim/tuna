export DATASET_NUM_PROC=1

dataset="hf-chat:heegyu/gretelai-synthetic-text-to-sql-trl
hf-chat:iknow-lab/spider-evol-1114-direct-trl
hf-chat:iknow-lab/spider-trl-241114
hf-chat:iknow-lab/bird-trl-241114"
model="infly/OpenCoder-1.5B-Base"
wandb online

run_name="opencoder-1.5b-sql"

train() {
    lr=$1
    python -m tuna.launcher.train \
        --do_train \
        --task chat-lm \
        --padding max_length \
        --model_arch causal-lm \
        --project "Text2SQL" \
        --train_template chatml \
        --run_name "$run_name" \
        --dataset="$dataset" \
        --load_from_cache_file \
        --packing False \
        --train_only_response True \
        --max_length=2048 \
        --amp \
        --lr_warmup_ratio 0.01 \
        --truncation \
        --model_name_or_path $model \
        --trust_remote_code \
        --total_epochs 3 \
        --learning_rate $lr \
        --train_total_batch_size 128 \
        --train_batch_size_per_device 2 \
        --eval_batch_size_per_device 2 \
        --save_strategy epoch \
        --push_to_hub \
        --revision_prefix "lr$lr-bs128" \
        --output_dir ""
}

train 1e-5
train 5e-6
train 1e-6