#! /bin/zsh

PROMPTS_DIR=${1}
OUT_DIR=${2}
MODEL=${3}
PROMPT_NAME=${4}

echo "running for ${MODEL}"
for fname in ${PROMPTS_DIR}/prompts*; do
    lang=${fname##*.}
    echo "processing $lang"

    out_name=${OUT_DIR}/responses_mc.$lang

    ## GPT-3
    # python run_gpt/run_inference.py -i $fname -o $out_name -b 1 -m $MODEL --sleep 0

    ## GPT-4
    ## expand max new tokens to 512, since the tokenizer needs more space in non Latin langs
    python run_gpt/run_inference.py -i $fname -o $out_name -b 1 -m $MODEL --sleep 0 \
    -sys run_gpt/system_prompts/${PROMPT_NAME}.txt -tok 512 -tm translate/terms

done
