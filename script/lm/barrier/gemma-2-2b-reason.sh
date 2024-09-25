wandb online
export LD_LIBRARY_PATH=$HOME/miniconda3/envs/tuna/lib/
model="google/gemma-2-2b-it"

eval() {
    type=$1
    lr=$2

    datasets="heegyu/wildjailbreak-train-reason10k-0917:$type"
    run_name="0925-gemma-2-2b-it-barrier-lora-$type-lr$lr"

    python -m tuna.launcher.train \
        --do_train \
        --task chat-lm \
        --padding max_length \
        --model_arch causal-lm \
        --project "llama-barrier" \
        --train_template gemma \
        --run_name "$run_name" \
        --dataset="$datasets" \
        --use_lora \
        --max_length=1024 \
        --model_name_or_path $model \
        --total_epochs 3 \
        --learning_rate $lr \
        --train_total_batch_size 32 \
        --train_batch_size_per_device 4 \
        --eval_batch_size_per_device 4 \
        --save_strategy epoch \
        --push_to_hub \
        --output_dir ""
}

eval reason 1e-5
eval reason 2e-5
eval reason 5e-5
eval reason 1e-4