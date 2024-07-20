wandb online

datasets="GAIR/lima,changpt/ko-lima-vicuna"
model="mistralai/Mistral-Nemo-Base-2407"
run_name="Mistral-Nemo-LIMA"

# accelerate launch -m tuna.launcher.train \
python -m tuna.launcher.train \
    --do_train \
    --mesh sp \
    --task chat-lm \
    --padding max_length \
    --model_arch causal-lm \
    --project "kor-llm" \
    --run_name "$run_name" \
    --dataset="$datasets" \
    --use_lora \
    --packing False \
    --max_length=1536 \
    --amp \
    --model_name_or_path $model \
    --logging_steps 1 \
    --total_epochs 3 \
    --learning_rate 1e-4 \
    --train_total_batch_size 32 \
    --train_batch_size_per_device 1 \
    --eval_batch_size_per_device 1 \
    --save_strategy last \
    --push_to_hub \
    --output_dir ""