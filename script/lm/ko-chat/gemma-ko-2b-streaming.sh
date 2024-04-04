export CUDA_VISIBLE_DEVICES=0

wandb online

# datasets="maywell/koVast,heegyu/PKU-SafeRLHF-ko:safer,FreedomIntelligence/evol-instruct-korean"
# run_name="gemma-ko-2b-0403"
datasets="hf-chat:heegyu/ko-openchat-0404"
run_name="gemma-ko-2b-0404"
model="beomi/gemma-ko-2b"

# accelerate launch -m tuna.launcher.train \
python -m tuna.launcher.train \
    --do_train --do_eval \
    --task chat-lm \
    --padding max_length \
    --model_arch causal-lm \
    --project "ko-openchat-2b" \
    --train_template gemma \
    --run_name "$run_name" \
    --dataset="$datasets" \
    --dataset_streaming \
    --packing \
    --amp \
    --max_length=2048 \
    --model_name_or_path $model \
    --learning_rate 5e-6 \
    --train_total_batch_size 32 \
    --train_batch_size_per_device 2 \
    --eval_batch_size_per_device 2 \
    --save_strategy steps \
    --eval_strategy steps \
    --total_steps 3000000 \
    --save_steps 150000 \
    --eval_steps 150000 \
    --push_to_hub \
    --output_dir ""