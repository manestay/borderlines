# BorderLines Dataset of Territorial Disputes

Code + Data for the [arXiv](https://arxiv.org/abs/2305.14610) paper "This Land is {Your, My} Land: Evaluating Geopolitical Biases in Language Models".

## I. Using BorderLines Dataset
In this repository, we include the data files for the default version of BorderLines (2023-05-15), which is based on the [2023-05-15](https://en.wikipedia.org/w/index.php?title=List_of_territorial_disputes&oldid=1154894956) article.

The files are:
* `disputed_territories.csv`: the main BorderLines territorial dispute table
* `countries_info.json`: demographic info for each country
* `prompts/prompts_q_mc.txt`: multiple-choice questions, in English, for each disputed territory
* `translate/prompts_q_mc/`: questions in multiple languages. For example `prompts.es` contains
  the questions, in Spanish, for disputed territories in which a Spanish-speaking country is involved

## II. Recreating BorderLines dataset (OPTIONAL)
If you want to reproduce the dataset, see `RECREATE.md`. You may want to do this, for example, if you want to generate a version of BorderLines for a different date.
Otherwise, skip to III.

Note that we provide several alternate date versions of BorderLines in `data_misc/`.

## III. Evaluation Suite on BorderLines

### 1. Run inference for language models
**NOTE**: The below commands run on BorderLines v1. If you followed section II. to create a new version, you should adjust the arguments accordingly.

We use rank classification to obtain a model's responses.

#### A. GPT-3 inference
```
# runs prompts on davinci
# run over English
python run_gpt/run_inference_rank.py -i prompts/prompts_q_mc.txt -o outputs/gpt3_dv/responses_mc_all.txt -c prompts/choices.txt -b 70 -m text-davinci-003 --print -k {OPENAI_API_KEY}

# run over languages
run_gpt/run_inference_rank_folder.sh translate/prompts_mc_q outputs/gpt3_dv text-davinci-003

# run prompts on curie
python run_gpt/run_inference_rank.py -i prompts/prompts_q_mc.txt -o outputs/gpt3_c/responses_mc_all.txt -c prompts/choices.txt -b 70 -m text-curie-001 --print -k {OPENAI_API_KEY}

run_gpt/run_inference_rank_folder.sh translate/prompts_mc_q outputs/gpt3_c text-curie-001

```
Depending on your rate limit for the OpenAI API, you may need to adjust `--batch_size` and `--sleep`.

#### B. GPT-4 inference
For GPT-4, we allow the model to generate a response, then parse a selection from the output. This parsing approach allows us to perform our prompt modification experiments.

We run for 4 system prompts; these are found in `run_gpt/system_prompts/{prompt_name}.txt`. Below we use the `vanilla` prompt, and you should modify as needed.

```
# run vanilla prompt on gpt-4-0613
# run over English
python run_gpt/run_inference.py -i prompts/prompts_q_mc.txt -o outputs/gpt4-0314/vanilla/responses_mc_all.txt -m gpt-4-0613 -sys run_gpt/system_prompts/vanilla.txt -tok 128

# run over languages
run_gpt/run_inference_rank_folder.sh translate/prompts_mc_q outputs/gpt4-0613/vanilla gpt-4-0613 vanilla
```

#### C. BLOOM inference
For BLOOM models:

```
# run over entire BorderLines dataset, in English
python rank_outputs/main.py prompts/prompts_q_mc.txt outputs/bloom/responses_mc_all.txt -m bigscience/bloom-7b1
# run with multilingual prompts
python rank_outputs/main.py translate/prompts_mc_q outputs/bloom-tai -m bigscience/bloom-7b1
```

#### 2. Evaluate!
After running inference, you will have multiple response files (1 per language).
Combine them into a response table by running:

```
python gen_response_table.py translate/terms outputs/gpt3_dv/ outputs/gpt3_dv/response_table.csv translate/prompts_mc_q/

python gen_response_table.py translate/terms outputs/gpt3_c/ outputs/gpt3_c/response_table.csv translate/prompts_mc_q/

# update the args to run for the other models and prompt settings
```

__Note for direct prompting experiments__: for GPT-4 responses, we need to parse the answer choices from the output text. The `gen_response_table.py` script will attempt to automatically parse at first. For responses where this fails, the script will ask the user to "Make a choice". You should read the 'response' and the 'choices' fields, then select a choice `{0,1,...}`.

#### 3. Analyze concurrence scores
Calculate the CS scores, as seen in Table 2 of the paper:
```
python -i calculate_CS.py outputs/gpt3_dv/response_table.csv

python -i calculate_CS.py outputs/gpt3_c/response_table.csv
```

## Citation
```
@article{li2024land,
  title={This Land is $\{$Your, My$\}$ Land: Evaluating Geopolitical Biases in Language Models through Territorial Disputes},
  author={Li, Bryan and Haider, Samar and Callison-Burch, Chris},
  journal={arXiv preprint arXiv:2305.14610},
  year={2024}
}
```
