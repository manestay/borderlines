#! /bin/zsh

PROMPTS_DIR=${1}
OUT_DIR=${2}
MODEL=${3}
USE_CHOICES=${4:-true}

echo "running for ${MODEL}"
for fname in ${PROMPTS_DIR}/prompts*; do
    lang=${fname##*.}
    echo "processing $|lang|"

    out_name=${OUT_DIR}/responses_mc.$lang

    if [ "${USE_CHOICES}" = true ]; then
        python run_gpt/run_inference_rank.py -i $fname -o $out_name \
        -c ${PROMPTS_DIR}/choices.$lang -b 20 -m $MODEL --print
    else
        python run_gpt/run_inference_rank.py -i $fname -o $out_name \
        $CHOICE_ARGS -b 20 -m $MODEL --print
    fi

done
