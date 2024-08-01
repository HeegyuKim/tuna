wandb online
model="google/gemma-2-9b-it"
template="gemma"

train() {
    lr=$1
    datasets=$2
    run_name=$3

    python -m tuna.launcher.train_flax \
        --mesh sp \
        --do_train \
        --task chat-lm \
        --padding max_length \
        --project "KoChat-SFT" \
        --run_name "$run_name-lr$lr" \
        --dataset="$datasets" \
        --packing \
        --packing_strategy pad \
        --truncation \
        --max_length=2048 \
        --model_name_or_path $model \
        --total_epochs 2 \
        --logging_steps 2048 \
        --learning_rate $lr \
        --last_learning_rate_ratio 0.3 \
        --lr_warmup_ratio 0.01 \
        --lr_scheduler cosine \
        --load_from_cache_file \
        --train_template $template \
        --train_total_batch_size 128 \
        --train_batch_size_per_device 1 \
        --eval_batch_size_per_device 1 \
        --push_to_hub \
        --push_to_hub_id $run_name \
        --save_strategy epoch \
        --save_epochs 2 \
        --revision_prefix "lr$lr-" \
        --output_dir "/data/checkpoint/$run_name"
}

datasets="
Magpie-Align/Magpie-Pro-MT-300K-v0.1
arcee-ai/infini-instruct-top-500k
AI-MO/NuminaMath-CoT
AI-MO/NuminaMath-TIR
iknow-lab/qarv-instruct-ko-mt-deduped
sft:heegyu/orca-math-korean-preference-cleaned:hard
HAERAE-HUB/K2-Feedback:score5
"
# heegyu/HRC
# jojo0217/korean_safe_conversation
# iknow-lab/ko-evol-writing-wiki
# CarrotAI/ko-instruction-dataset
# maywell/kiqu_samples

TODAY=$(date +%m%d)

train 2e-5 "$datasets" "$TODAY-gemma2-koenzh"