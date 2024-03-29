export CUDA_VISIBLE_DEVICES=0

wandb online

# datasets="kyujinpy/KOR-OpenOrca-Platypus-v3,nampdn-ai/tiny-codes,nvidia/OpenMathInstruct-1"
datasets="kyujinpy/KOR-OpenOrca-Platypus-v3"
model="google/gemma-2b-it"
run_name="gemma-2b-it-kor-openorca-platypus-v3"

# accelerate launch -m tuna.launcher.train \
python -m tuna.launcher.train \
    --do_train \
    --task chat-lm \
    --padding max_length \
    --model_arch causal-lm \
    --project "kor-llm" \
    --run_name "$run_name" \
    --dataset="$datasets" \
    --packing \
    --max_length=2048 \
    --model_name_or_path $model \
    --logging_steps 1 \
    --total_epochs 3 \
    --use_lora \
    --limit 1024 \
    --learning_rate 1e-5 \
    --train_total_batch_size 32 \
    --train_batch_size_per_device 4 \
    --eval_batch_size_per_device 4 \
    --save_strategy epoch \
    --push_to_hub \
    --output_dir ""