wandb offline
model="prajjwal1/bert-tiny"
# model="google-bert/bert-base-uncased"
# dataset="dair-ai/emotion"
dataset="augesc:context"


python -m tuna.launcher.train \
    --mesh dp \
    --do_train --do_eval \
    --padding max_length \
    --truncation \
    --truncation_side left \
    --num_labels 8 \
    --task sequence-classification \
    --model_arch sequence-classification \
    --project "test" \
    --run_name "test" \
    --dataset="$dataset" \
    --max_length=64 \
    --model_name_or_path $model \
    --logging_steps 8 \
    --total_epochs 100 \
    --limit 1024 \
    --learning_rate 1e-4 \
    --train_total_batch_size 32 \
    --train_batch_size_per_device 32 \
    --eval_batch_size_per_device 32 \
    --save_strategy no \
    --output_dir ""