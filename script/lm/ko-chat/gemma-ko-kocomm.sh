export CUDA_VISIBLE_DEVICES=0

wandb online

# datasets="maywell/koVast,heegyu/PKU-SafeRLHF-ko:safer,FreedomIntelligence/evol-instruct-korean"
# run_name="gemma-ko-2b-0403"
datasets="hf-chat:heegyu/ko-openchat-0404"
run_name="gemma-ko-2b-kocomm-0403"
model="beomi/gemma-ko-2b"

# accelerate launch -m tuna.launcher.train \
python -m tuna.launcher.train \
    --do_train \
    --task chat-lm \
    --padding max_length \
    --model_arch causal-lm \
    --project "ko-openchat-2b" \
    --train_template gemma \
    --run_name "$run_name" \
    --dataset="$datasets" \
    --packing \
    --dataset_streaming \
    --amp \
    --max_length=2048 \
    --truncation \
    --truncation_side left \
    --model_name_or_path $model \
    --total_steps 1450000 \
    --learning_rate 1e-5 \
    --train_total_batch_size 32 \
    --train_batch_size_per_device 2 \
    --eval_batch_size_per_device 2 \
    --save_strategy steps \
    --save_steps 200000 \
    --push_to_hub \
    --output_dir ""