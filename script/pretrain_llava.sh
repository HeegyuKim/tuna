export CUDA_VISIBLE_DEVICES=1

wandb online

model="42dot/42dot_LLM-SFT-1.3B"

accelerate launch -m tuna.launcher.train \
    --do_train \
    --task llava-pretrain \
    --model_arch llava-for-pretrain \
    --project "llava" \
    --run_name "42dot_LLM-kollava-pretrain" \
    --dataset="kollava-pretrain" \
    --train_only_response False \
    --max_length=1024 \
    --vision_tower openai/clip-vit-large-patch14 \
    --model_name_or_path $model \
    --train_template $model \
    --logging_steps 32 \
    --total_epochs 1 \
    --learning_rate 2e-5 \
    --weight_decay 0 \
    --train_total_batch_size 256 \
    --train_batch_size_per_device 4 \
    --eval_batch_size_per_device 4 \
    --save_strategy epoch \
    --push_to_hub \
    --output_dir ""