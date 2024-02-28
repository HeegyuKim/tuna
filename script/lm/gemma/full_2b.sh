export CUDA_VISIBLE_DEVICES=0

wandb offline
model="google/gemma-2b-it"

train() {
    datasets=$1
    run_name=$2

    python -m tuna.launcher.train \
        --do_train \
        --task chat-lm \
        --padding max_length \
        --model_arch causal-lm \
        --project "kor-llm" \
        --run_name "$run_name" \
        --dataset="$datasets" \
        --packing \
        --use_lora \
        --max_length=1024 \
        --model_name_or_path $model \
        --total_epochs 3 \
        --learning_rate 1e-5 \
        --train_total_batch_size 128 \
        --train_batch_size_per_device 4 \
        --eval_batch_size_per_device 4 \
        --save_strategy epoch \
        --push_to_hub \
        --output_dir ""
}

train "kyujinpy/KOR-OpenOrca-Platypus-v3" "full-gemma-2b-it-kor-openorca-platypus-v3"
train "heegyu/OpenOrca-gugugo-ko-len500" "full-gemma-2b-it-openorca-gugugo-ko-len500"
train "heegyu/PKU-SafeRLHF-ko:safer" "full-gemma-2b-it-pku-saferlhf-ko-safer"