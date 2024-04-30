
model="heegyu/KoPrometheus-8B-0427@epoch-1"

check() {
    target=$1
    python -m eval.koprometheus --judge_model "heegyu/KoPrometheus-8B-0427@epoch-1" --target $target
}

check "HyperClovaX"
check "gpt-3.5-turbo-0125"
check "nlpai-lab_KULLM3"
check "solar-1-mini-chat"
