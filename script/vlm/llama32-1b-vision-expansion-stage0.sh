
dataset="heegyu/llava-pretrain-titok-256px"
# model="meta-llama/Llama-3.2-3B-Instruct"
# run_name="llama-3.2-3B-Instruct-viex-titok-1120"

model="meta-llama/Llama-3.1-8B-Instruct"
run_name="llama-3.1-8B-Instruct-viex-titok-1120"

wandb online


train() {
    lr=$1
    python -m tuna.launcher.train \
        --do_train \
        --task chat-lm \
        --padding max_length \
        --model_arch causal-lm-vision-expansion-stage1 \
        --project "Any2Any" \
        --train_template llama3 \
        --num_visual_tokens 4098 \
        --run_name "$run_name-lr$lr-bs128" \
        --dataset="$dataset" \
        --check_dataset False \
        --load_from_cache_file \
        --packing False \
        --train_only_response True \
        --max_length=256 \
        --amp True \
        --lr_warmup_ratio 0.01 \
        --truncation \
        --model_name_or_path $model \
        --total_epochs 2 \
        --learning_rate $lr \
        --train_total_batch_size 128 \
        --train_batch_size_per_device 4 \
        --eval_batch_size_per_device 4 \
        --save_strategy epoch \
        --push_to_hub False \
        --output_dir "/data/checkpoint/llama31-8b-titok-256px-stage1/"
}

train 5e-5
train 1e-4
# train 2e-4
train 3e-4
# train 6e-4