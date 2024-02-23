export CUDA_VISIBLE_DEVICES=1
export HF_DATASETS_CACHE=/data/.cache
export USE_TORCH=True

wandb online

model="google/gemma-2b-it"
image_prefix="<bos> [Image]"

python -m tuna.launcher.train \
    --mesh mp \
    --do_train \
    --task llava-pretrain \
    --model_arch llava-for-pretrain \
    --project "llava" \
    --run_name "gemma-2b-it-kollava-pretrain" \
    --dataset="kollava-pretrain" \
    --train_template gemma-vision \
    --image_prefix "$image_prefix" \
    --max_length=1024 \
    --vision_tower openai/clip-vit-large-patch14 \
    --model_name_or_path $model \
    --train_template $model \
    --logging_steps 32 \
    --total_epochs 2 \
    --learning_rate 2e-5 \
    --weight_decay 0 \
    --train_total_batch_size 128 \
    --train_batch_size_per_device 8 \
    --eval_batch_size_per_device 8 \
    --save_strategy epoch \
    --push_to_hub \
    --output_dir ""
    