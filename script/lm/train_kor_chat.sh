export CUDA_VISIBLE_DEVICES=0

export HF_DATASETS_CACHE="/data/hf-datasets-cache/"

wandb online

datasets="kyujinpy/KOR-OpenOrca-Platypus-v3,nampdn-ai/tiny-codes,nvidia/OpenMathInstruct-1"

# accelerate launch -m tuna.launcher.train \
python -m tuna.launcher.train \
    --do_train \
    --task chat-lm \
    --padding max_length \
    --model_arch causal-lm \
    --project "kor-llm" \
    --run_name "42dot_sft_code_math" \
    --dataset="$datasets" \
    --packing \
    --max_length=2048 \
    --model_name_or_path 42dot/42dot_LLM-SFT-1.3B \
    --logging_steps 1 \
    --total_epochs 32 \
    --learning_rate 1e-5 \
    --train_total_batch_size 128 \
    --train_batch_size_per_device 4 \
    --eval_batch_size_per_device 4 \
    --save_strategy epoch \
    --output_dir ""