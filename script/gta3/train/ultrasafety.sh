wandb online
dataset="droussis/UltraSafety_binarized-orpo-dpo"

model="Nexusflow/Starling-LM-7B-beta"
template="openchat"

# run_name="0504-Meta-Llama-3-8B-Instruct-Gate$name-5e-5"
# model="meta-llama/Meta-Llama-3-8B-Instruct"
# template="llama3"

train() {
    dataset=$1
    task=$2
    run_name="0505-Starling-LM-7B-beta-UltraSafety-$task-5e-5"

    python -m tuna.launcher.train \
        --do_train \
        --task $task \
        --padding max_length \
        --model_arch causal-lm \
        --project "GTA3-GATE" \
        --train_template $template \
        --run_name "$run_name" \
        --dataset="$dataset" \
        --packing False \
        --amp \
        --use_lora \
        --max_length=2048 \
        --truncation \
        --model_name_or_path $model \
        --total_epochs 3 \
        --learning_rate 5e-5 \
        --train_total_batch_size 32 \
        --train_batch_size_per_device 1 \
        --eval_batch_size_per_device 1 \
        --save_strategy epoch \
        --push_to_hub \
        --output_dir ""
}

train "sft:droussis/UltraSafety_binarized-orpo-dpo" "sft"
train "dpo:droussis/UltraSafety_binarized-orpo-dpo" "dpo"
