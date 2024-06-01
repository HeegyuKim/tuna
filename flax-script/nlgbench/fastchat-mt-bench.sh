export HF_HOME="/data/hf-home"

eval() {
    model=$1
    python -m fastchat.llm_judge.gen_flax_answer --model "$model" --chat-template "zephyr" 
}

eval "heegyu/0507-zephyr-7b-sft-full-max-margin-1e-4-b0.01@epoch-1"
eval "heegyu/0507-zephyr-7b-sft-full-max-margin-1e-4-b0.1@epoch-1"
eval "heegyu/0507-zephyr-7b-sft-full-max-margin-1e-4-b0.5@epoch-1"

eval "heegyu/0510-dpo-7b@lr2e-4-beta0.01-epoch-1"
eval "heegyu/0510-dpo-7b@lr3e-4-beta0.01-epoch-1"
eval "heegyu/0510-dco-v2-7b@lr2e-4-beta0.01-epoch-1"