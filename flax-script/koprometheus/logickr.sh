
model="peft:heegyu/0708-ko-prometheus@epoch-1"

check() {
    target=$1
    python -m eval.koprometheus --judge_model "$model"
}

check "*"
