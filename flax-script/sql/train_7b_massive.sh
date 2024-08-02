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
        --train_template deepseek \
        --packing \
        --packing_strategy pad \
        --max_length=3072 \
        --truncation \
        --model_name_or_path $model \
        --total_epochs 2 \
        --lr_warmup_ratio 0.05 \
        --last_learning_rate_ratio 0.1 \
        --learning_rate 1e-5 \
        --train_total_batch_size 32 \
        --train_batch_size_per_device 1 \
        --eval_batch_size_per_device 1 \
        --save_strategy epoch \
        --push_to_hub \
        --output_dir ""
}

# MATH_MODEL="deepseek-ai/deepseek-math-7b-base"
# train $MATH_MODEL "math-7b-aimo-sql" "chinmayc3/bird-sql,xlangai/spider,AI-MO/NuminaMath-CoT,AI-MO/NuminaMath-TIR"

MATH_MODEL=mistralai/mathstral-7B-v0.1
train $MATH_MODEL "mistral-math-7b-aimo-sql" "chinmayc3/bird-sql,xlangai/spider,AI-MO/NuminaMath-CoT,AI-MO/NuminaMath-TIR"

# train $MATH_MODEL "math-7b-aimo-sql" "chinmayc3/bird-sql,xlangai/spider"

# CODE_MODEL="deepseek-ai/deepseek-coder-6.7b-base"
# train $CODE_MODEL "code-7b-spider" "xlangai/spider"