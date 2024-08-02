wandb online
export HF_HOME=/data/hf-home/


train() {
    model=$1
    name=$2
    dataset=$3

    run_name="deepseek-$name"

    # 3K seq len 가능

    python -m tuna.launcher.train_flax \
        --mesh sp \
        --do_train --do_eval \
        --task chat-lm \
        --padding max_length \
        --project "Text2SQL" \
        --run_name "$run_name" \
        --dataset="$dataset" \
        --train_template gemma \
        --packing False \
        --max_length=4096 \
        --truncation \
        --model_name_or_path $model \
        --total_epochs 3 \
        --lr_warmup_ratio 0.05 \
        --last_learning_rate_ratio 0.1 \
        --learning_rate 5e-6 \
        --train_total_batch_size 32 \
        --train_batch_size_per_device 1 \
        --eval_batch_size_per_device 1 \
        --save_strategy epoch \
        --push_to_hub false \
        --output_dir "/data/checkpoint/$run_name"
}

MODEL="google/gemma-2-2b-it"
train $MODEL "gemma-2-2b-spider" "xlangai/spider"
train $MODEL "gemma-2-2b-bird" "chinmayc3/bird-sql"