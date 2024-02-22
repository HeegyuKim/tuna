export CUDA_VISIBLE_DEVICES=1
export HF_DATASETS_CACHE=/data/.cache
export USE_TORCH=True

wandb offline

model="42dot/42dot_LLM-SFT-1.3B"

python -m tuna.launcher.train \
    --do_train \
    --task llava-finetune \
    --model_arch llava-for-finetune \
    --project "llava" \
    --run_name "42dot_LLM-kollava-finetune" \
    --dataset="kollava-finetune" \
    --max_length=1024 \
    --model_name_or_path $model \
    --revision epoch-0 \
    --train_template "42dot" \
    --logging_steps 32 \
    --total_epochs 3 \
    --learning_rate 2e-5 \
    --limit 1024 \
    --weight_decay 0 \
    --train_total_batch_size 128 \
    --train_batch_size_per_device 2 \
    --eval_batch_size_per_device 2 \
    --save_strategy epoch \
    --output_dir ""