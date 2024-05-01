wandb online
model="google/gemma-1.1-7b-it"
dataset="promtheus:heegyu/feedback-collection-ko-split"


python -m tuna.launcher.train_flax \
    --mesh sp \
    --do_train \
    --task chat-lm \
    --padding max_length \
    --project "KoChat-SFT" \
    --run_name "KoPrometheus-7B-0501" \
    --dataset="$dataset" \
    --packing False \
    --max_length=2048 \
    --truncation \
    --model_name_or_path $model \
    --logging_steps 1 \
    --total_epochs 3 \
    --learning_rate 2e-5 \
    --train_template gemma \
    --last_learning_rate_ratio 0.1 \
    --train_total_batch_size 32 \
    --train_batch_size_per_device 1 \
    --eval_batch_size_per_device 1 \
    --save_strategy epoch \
    --push_to_hub \
    --output_dir ""