wandb offline
model="prajjwal1/bert-tiny"
# model="google-bert/bert-base-uncased"

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
    --dataset="dair-ai/emotion" \
    --max_length=64 \
    --model_name_or_path $model \
    --logging_steps 4 \
    --total_epochs 10 \
    --learning_rate 1e-4 \
    --train_total_batch_size 8 \
    --train_batch_size_per_device 8 \
    --eval_batch_size_per_device 8 \
    --save_strategy no \
    --output_dir ""