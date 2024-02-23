export CUDA_VISIBLE_DEVICES=1
export HF_DATASETS_CACHE=/data/.cache
export USE_TORCH=True

wandb offline
model="42dot/42dot_LLM-SFT-1.3B"
# image_prefix="[Image]"
image_prefix=""

python -m tuna.launcher.train \
    --do_train \
    --task llava-pretrain \
    --model_arch llava-for-pretrain \
    --project "llava" \
    --run_name "42dot_LLM-kollava-pretrain" \
    --dataset="kollava-pretrain" \
    --max_length=512 \
    --vision_tower openai/clip-vit-large-patch14 \
    --model_name_or_path $model \
    --train_template $model \
    --logging_steps 32 \
    --total_epochs 2 \
    --learning_rate 1e-4 \
    --limit 1000 \
    --weight_decay 0 \
    --train_total_batch_size 128 \
    --train_batch_size_per_device 4 \
    --eval_batch_size_per_device 4 \
    --save_strategy no \
    --push_to_hub \
    --output_dir ""
    