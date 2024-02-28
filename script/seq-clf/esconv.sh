wandb online
model="FacebookAI/xlm-roberta-large"
# model="FacebookAI/xlm-roberta-base"

python -m tuna.launcher.train \
    --mesh dp \
    --do_train --do_eval \
    --padding max_length \
    --num_labels 8 \
    --truncation \
    --truncation_side left \
    --task sequence-classification \
    --model_arch sequence-classification \
    --project "esconv" \
    --run_name "esconv-xlm-roberta-large" \
    --dataset="esconv:context" \
    --max_length=512 \
    --model_name_or_path $model \
    --total_epochs 5 \
    --learning_rate 1e-4 \
    --train_total_batch_size 32 \
    --train_batch_size_per_device 32 \
    --eval_batch_size_per_device 32 \
    --save_strategy epoch \
    --save_epochs 1 \
    --push_to_hub \
    --output_dir ""