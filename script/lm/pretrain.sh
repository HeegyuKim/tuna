wandb offline

export HF_DATASETS_CACHE="/data/hf-datasets-cache/"

python -m tuna.launcher.train \
    --mesh dp \
    --do_train \
    --task pretrain-lm \
    --model_arch causal-lm \
    --project "test" \
    --run_name "test" \
    --dataset="nvidia/OpenMathInstruct-1" \
    --max_length=1024 \
    --model_name_or_path EleutherAI/pythia-160m \
    --logging_steps 1 \
    --total_epochs 3 \
    --limit 1000 \
    --learning_rate 1e-5 \
    --train_total_batch_size 128 \
    --train_batch_size_per_device 8 \
    --eval_batch_size_per_device 8 \
    --save_strategy no \
    --output_dir ""