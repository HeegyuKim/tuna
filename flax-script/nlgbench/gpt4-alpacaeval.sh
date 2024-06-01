# export OPENAI_API_KEY=<your_api_key> # for more complex configs, e.g. using Azure or switching clients see client_configs/README.md 
eval() {
    model=$1
    alpaca_eval --model_outputs "outputs/$model/alpaca-eval.json"
}

# eval "alignment-handbook/zephyr-7b-sft-full" # 8.48      5.12
# eval "alignment-handbook/zephyr-7b-dpo-full" # 16.58     13.77
# eval "HuggingFaceH4/zephyr-7b-beta"  # 11.85     10.15

# eval "heegyu/0510-dco-v2-7b@lr2e-4-beta0.01-epoch-1" 10.95
# eval "heegyu/0510-dco-v4-7b@lr2e-4-beta0.01-epoch-1"  11.22      8.98 

# eval "heegyu/0507-zephyr-7b-sft-full-max-margin-1e-4-b0.01@epoch-1" # 11.39      9.47
# eval "heegyu/0507-zephyr-7b-sft-full-max-margin-1e-4-b0.1@epoch-1" #  10.54      7.39
# eval "heegyu/0507-zephyr-7b-sft-full-max-margin-1e-4-b0.5@epoch-1" #  9.39      5.57 