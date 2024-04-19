export CUDA_VISIBLE_DEVICES=0

wandb offline

# datasets="kyujinpy/KOR-OpenOrca-Platypus-v3,nampdn-ai/tiny-codes,nvidia/OpenMathInstruct-1"
datasets="kyujinpy/KOR-OpenOrca-Platypus-v3"
model="google/gemma-7b-it"
run_name="gemma-7b-it-kor-openorca-platypus-v3"

# accelerate launch -m tuna.launcher.train \
python -m tuna.launcher.train \
    --mesh mp \
    --do_train \
    --task chat-lm \
    --padding max_length \
    --model_arch causal-lm \
    --project "kor-llm" \
    --run_name "$run_name" \
    --dataset="$datasets" \
    --packing \
    --amp \
    --max_length=1024 \
    --model_name_or_path $model \
    --total_epochs 3 \
    --learning_rate 1e-5 \
    --train_total_batch_size 32 \
    --train_batch_size_per_device 1 \
    --eval_batch_size_per_device 1 \
    --save_strategy epoch \
    --push_to_hub \
    --output_dir ""