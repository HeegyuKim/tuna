wandb online

train() {
    dataset=$1
    name=$2

    run_name="0507-Starling-LM-7B-beta-Gate$name-5e-5"
    model="Nexusflow/Starling-LM-7B-beta"
    template="openchat"

    # run_name="0507-Meta-Llama-3-8B-Instruct-LoraGuard$name-5e-5"
    # model="meta-llama/Meta-Llama-3-8B-Instruct"
    # template="llama3"

    python -m tuna.launcher.train \
        --do_train \
        --task chat-lm \
        --padding max_length \
        --model_arch causal-lm \
        --project "GTA3-GATE" \
        --train_template $template \
        --run_name "$run_name" \
        --dataset="$dataset" \
        --packing False \
        --amp \
        --use_lora \
        --max_length=1024 \
        --truncation \
        --model_name_or_path $model \
        --total_epochs 3 \
        --learning_rate 5e-5 \
        --train_total_batch_size 32 \
        --train_batch_size_per_device 2 \
        --eval_batch_size_per_device 2 \
        --save_strategy epoch \
        --push_to_hub \
        --output_dir ""
}

train "llamaguard:iknow-lab/GTA3-promptguard-v0507" ""
train "llamaguard-cot:iknow-lab/GTA3-promptguard-v0507" "-cot"