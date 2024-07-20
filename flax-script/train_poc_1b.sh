wandb online
# export HF_HOME=/data-plm/hf-home
# model="TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
model="mistralai/Mistral-Nemo-Base-2407"
# model="HuggingFaceM4/tiny-random-LlamaForCausalLM"

python -m tuna.launcher.train_flax \
    --mesh sp \
    --do_train \
    --task chat-lm \
    --padding max_length \
    --project "test" \
    --run_name "mistral-nemo-12b-lima" \
    --dataset="GAIR/lima,changpt/ko-lima-vicuna" \
    --train_template llama \
    --packing False \
    --max_length=1536 \
    --truncation \
    --model_name_or_path $model \
    --total_epochs 3 \
    --lr_warmup_ratio 0.05 \
    --last_learning_rate 0.1 \
    --learning_rate 5e-5 \
    --train_total_batch_size 32 \
    --train_batch_size_per_device 1 \
    --eval_batch_size_per_device 1 \
    --save_strategy no \
    --push_to_hub \
    --output_dir ""