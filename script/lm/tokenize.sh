export HF_DATASETS_CACHE="/data/hf-datasets-cache/"

datasets="tatsu-lab/alpaca,HuggingFaceH4/ultrachat_200k"

python -m tuna.launcher.pretrain_tokenize \
    --dataset="$datasets" \
    --max_length=1024 \
    --tokenizer EleutherAI/pythia-160m \
    --output_dir "/data/tokenized/test"