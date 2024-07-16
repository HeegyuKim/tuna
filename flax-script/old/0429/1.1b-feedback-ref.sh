wandb online
model="heegyu/TinyLlama__TinyLlama-1.1B-intermediate-step-1431k-3T-tinyllama-1.1b-sft@steps-155897"

python -m tuna.launcher.train_flax \
    --mesh fsdp \
    --do_train \
    --task chat-lm \
    --padding max_length \
    --project "DDFO-SFT" \
    --run_name "TinyLlama-Feedback-Ref" \
    --dataset="dfo:kaist-ai/Feedback-Collection" \
    --packing False \
    --max_length=1024 \
    --truncation \
    --train_template zephyr \
    --model_name_or_path $model \
    --total_epochs 3 \
    --last_learning_rate_ratio 0.1 \
    --lr_warmup_ratio 0.1 \
    --learning_rate 5e-5 \
    --train_total_batch_size 32 \
    --train_batch_size_per_device 4 \
    --eval_batch_size_per_device 4 \
    --save_strategy epoch \
    --push_to_hub \
    --output_dir ""