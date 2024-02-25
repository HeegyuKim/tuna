wandb online
model="FacebookAI/xlm-roberta-large"

python -m tuna.launcher.train \
    --mesh fsdp \
    --do_train --do_eval \
    --padding max_length \
    --num_labels 8 \
    --truncation \
    --truncation_side left \
    --task sequence-classification \
    --model_arch sequence-classification \
    --project "esconv" \
    --run_name "augesc-xlm-roberta-large" \
    --dataset="augesc:context" \
    --max_length=512 \
    --model_name_or_path $model \
    --total_epochs 10 \
    --learning_rate 1e-4 \
    --train_total_batch_size 32 \
    --train_batch_size_per_device 8 \
    --eval_batch_size_per_device 8 \
    --save_strategy epoch \
    --save_epochs 5 \
    --push_to_hub \
    --output_dir ""