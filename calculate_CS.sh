#!/bin/bash
# set -x

TABLE_PATH=all_results/scores_for_all.csv

for csv_path in all_results/*.csv; do
    if [ $csv_path == "$TABLE_PATH" ]; then
        continue # skip the final table file
    fi
    score_path="${csv_path%.csv}_scores.json"
    if [ -f "$score_path" ]; then
        echo "Score file $score_path already exists. Skipping..."
        continue # comment this line to recompute all scores
    else
        echo "Calculating scores for $csv_path..."
    fi
    python ./calculate_CS.py "$csv_path" -o "$score_path" -q
done

echo "All metrics calculated, generating table at ${TABLE_PATH}..."
python ./gen_all_results_table.py \
    -i all_results/*scores.json \
    -o ${TABLE_PATH} --print_latex
