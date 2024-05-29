wandb online

template="gemma"
lr=2e-5

train() {
    subject=$1
    dataset="iknow-lab/mmlu-test-50to50-$subject"
    task=$2
    model="google/gemma-$3"

    hub_id="0530-gemma-$3-mmlu-toy-sft-flax"

    python -m tuna.launcher.train_flax \
        --mesh sp \
        --do_train --do_eval \
        --task $task \
        --padding max_length \
        --project "GTA3-TOY" \
        --train_template $template \
        --run_name "$hub_id-$subject" \
        --dataset="$dataset" \
        --packing False \
        --max_length=1024 \
        --truncation \
        --model_name_or_path $model \
        --total_epochs 1 \
        --learning_rate $lr \
        --lr_warmup_ratio 0.1 \
        --train_total_batch_size 32 \
        --train_batch_size_per_device 1 \
        --eval_batch_size_per_device 1 \
        --save_strategy epoch \
        --push_to_hub \
        --push_to_hub_id $hub_id \
        --revision_prefix "$subject-$lr" \
        --output_dir ""
}

size=$1
# size must be 2b or 7b
if [ "$size" != "2b" ] && [ "$size" != "7b" ]; then
    echo "size must be 2b or 7b"
    exit 1
fi

train "base" "chat-lm" $size
train "stem" "chat-lm" $size
train "social_sciences" "chat-lm" $size
train "humanities" "chat-lm" $size
train "other" "chat-lm" $size

train "stem_aug" "chat-lm" $size
train "social_sciences_aug" "chat-lm" $size
train "humanities_aug" "chat-lm" $size
train "other_aug" "chat-lm" $size

# for subject_pre in "abstract_algebra" "human_sexuality" "moral_disputes" "high_school_us_history"; do
#     train $subject_pre "chat-lm" 8 $size
#     train "$subject_pre-aug" "chat-lm" 8 $size
# done
