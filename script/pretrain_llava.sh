export CUDA_VISIBLE_DEVICES=0

wandb online

accelerate launch -m tuna.launcher.train \
    --do_train \
    --task chat-vlm \
    --model_arch llava-for-pretrain \
    --project "llava" \
    --run_name "TinyMistral-248M-v2.5-Instruct-VQA" \
    --dataset="BilbaoQA" \
    --train_only_response False \
    --max_length=1024 \
    --vision_tower openai/clip-vit-base-patch32 \
    --model_name_or_path Locutusque/TinyMistral-248M-v2.5-Instruct \
    --train_template default \
    --logging_steps 1 \
    --total_epochs 3 \
    --learning_rate 1e-5 \
    --weight_decay 0 \
    --train_total_batch_size 32 \
    --train_batch_size_per_device 1 \
    --eval_batch_size_per_device 1 \
    --save_strategy epoch \
    --push_to_hub \
    --output_dir ""