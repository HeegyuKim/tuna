wandb online
model="Qwen/Qwen2-7B"

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
        --packing True \
        --truncation \
        --max_length=2048 \
        --model_name_or_path $model \
        --total_epochs 2 \
        --learning_rate $lr \
        --last_learning_rate_ratio 0.3 \
        --lr_warmup_ratio 0.01 \
        --lr_scheduler cosine \
        --adam_beta2 0.95 \
        --load_from_cache_file \
        --train_template chatml \
        --train_total_batch_size 32 \
        --train_batch_size_per_device 1 \
        --eval_batch_size_per_device 1 \
        --push_to_hub \
        --push_to_hub_id $run_name \
        --save_strategy epoch \
        --revision_prefix "lr$lr-" \
        --output_dir "/data/checkpoint/$run_name"
}

train 2e-5 "Magpie-Align/Magpie-Qwen2-Pro-300K-Filtered,Magpie-Align/Magpie-Pro-MT-300K-v0.1,HAERAE-HUB/qarv-instruct-100k,jojo0217/korean_safe_conversation,heegyu/HRC,iknow-lab/ko-genstruct-v1-output:simple,CarrotAI/ko-instruction-dataset,maywell/kiqu_samples,HAERAE-HUB/K2-Feedback:score5" "0713-qwen2-koenzh"