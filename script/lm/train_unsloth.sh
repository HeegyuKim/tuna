export CUDA_VISIBLE_DEVICES=1
export WANDB_PROJECT="test"


# see https://huggingface.co/docs/transformers/v4.42.0/en/main_classes/trainer#transformers.TrainingArguments

wandb offline
model="unsloth/Mistral-Nemo-Base-2407-bnb-4bit"
# model="unsloth/Phi-3-mini-4k-instruct"

# accelerate launch -m tuna.launcher.train \
python -m tuna.launcher.train_unsloth \
    --max_length=512 \
    --model_name_or_path $model \
    --dataset "GAIR/lima" \
    --train_template llama2 \
    --logging_steps 1 \
    --num_train_epochs 3 \
    --use_lora \
    --optim adamw_8bit \
    --lr_scheduler_type linear \
    --learning_rate 1e-5 \
    --gradient_accumulation_steps 32 \
    --per_device_train_batch_size 1 \
    --warmup_steps 8 \
    --save_strategy epoch \
    --load_in_4bit \
    --output_dir "unsloth_outputs/"