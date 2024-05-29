wandb online

template="gemma"

train() {
    subject=$1
    dataset="iknow-lab/mmlu-test-100-$subject"
    task=$2
    lora_r=$3
    model="google/gemma-1.1-$4-it"

    hub_id="0528-gemma-$4-mmlu-toy-sft-lora"

    python -m tuna.launcher.train \
        --mesh sp \
        --do_train --do_eval \
        --task $task \
        --padding max_length \
        --model_arch causal-lm \
        --project "GTA3-TOY" \
        --train_template $template \
        --run_name "$hub_id-r$lora_r-$subject" \
        --dataset="$dataset" \
        --packing False \
        --trust_remote_code \
        --amp True \
        --use_lora $use_lora \
        --lora_r $lora_r \
        --lora_alpha $((lora_r * 2)) \
        --max_length=1024 \
        --truncation \
        --model_name_or_path $model \
        --total_epochs 3 \
        --learning_rate 5e-5 \
        --lr_warmup_ratio 0.1 \
        --train_total_batch_size 32 \
        --train_batch_size_per_device 2 \
        --eval_batch_size_per_device 2 \
        --save_strategy epoch \
        --push_to_hub \
        --push_to_hub_id $hub_id \
        --revision_prefix "$subject-r$lora_r-5e-5" \
        --output_dir ""
}

size=$1
# size must be 2b or 7b
if [ "$size" != "2b" ] && [ "$size" != "7b" ]; then
    echo "size must be 2b or 7b"
    exit 1
fi

for subject in "abstract_algebra" "human_sexuality" "moral_disputes" "high_school_us_history"; do
    train "$subject-aug" "chat-lm" 8 $size
    train $subject "chat-lm" 8 $size
done
