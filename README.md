# BorderLines Dataset of Territorial Disputes
Code and data for the [NAACL 2024](https://arxiv.org/abs/2305.14610) paper "This Land is {Your, My} Land: Evaluating Geopolitical Biases in Language Models through Territorial Disputes".

## I. Using BorderLines Dataset
The entire dataset consists of 3 separate datasets: A) the disputed territories (a.k.a. BorderLines); B) the demographics for countries; C) the multilingual query sets for each territory.

You can obtain the dataset by running either option 1, loading from the datasets hub, or option 2, cloning the repository.

### 1. Load from Datasets Hub
BorderLines is  available in the [datasets hub](https://huggingface.co/datasets/manestay/borderlines). Load by running:

```
import datasets

# load disputed territories
territories = datasets.load_dataset('manestay/borderlines', 'territories')['train']

# load country demographics
countries = datasets.load_dataset('manestay/borderlines', 'countries')['train']

# load queries in 49 languages
queries = datasets.load_dataset('manestay/borderlines', 'queries')
```
Note: the above code is included in the function `load_borderlines_hf` of file `run_gpt/lib.py`.

### 2. Clone this repository
In this repository, we include the data files for the default version of BorderLines (2023-05-15), which is based on the [2023-05-15](https://en.wikipedia.org/w/index.php?title=List_of_territorial_disputes&oldid=1154894956) article.

The files are:
* `disputed_territories.csv`: the main BorderLines territorial dispute table
* `countries_info.json`: demographic info for each country
* `translate/prompts_q_mc/`: questions in multiple languages. For example `prompts.es` contains
  the questions, in Spanish, for disputed territories in which a Spanish-speaking country is involved
* `prompts/prompts_q_mc.txt`: multiple-choice questions, in English, for each disputed territory. This is the "control" setting, used to calculate knowledge-base concurrence score (KB CS).

#### Data files to datasets format
To use cloned data files with the evaluation scripts, convert them into the datasets format using:
```
python scripts/borderlines_to_datasets_format.py -o datasets/v1 -p prompts/prompts_q_mc.txt -td translate/terms -tp disputed_territories.csv -ip countries_info.json -pd translate/prompts_mc_q
```

## II. Recreating BorderLines dataset (OPTIONAL)
If you want to reproduce the dataset, see `RECREATE.md`. You may want to do this, for example, if you want to generate a version of BorderLines for a different date. Otherwise, skip to III.

Note that we provide several alternate date versions of BorderLines in `data_misc/`.

## III. Evaluation Suite on BorderLines

### 1. Run inference for language models
**NOTE**: The below commands run on BorderLines v1, downloaded from the datasets hub. If you are running on a local version (i.e. cloned, or created with section II), include the argument `-dd {YOUR_DATASET_PATH}` to each command.

#### A. GPT-3 inference
For GPT-3 models, we use rank classification. This means that given a query, and choices A and B, we concatenate each choice {query + A, query + B}, calculate the probability of either prompt, and assign the more likely one as the model's response.

__NOTE__: As of 2024/01/04, OpenAI has deprecated `text-davinci-003` and the other Completion endpoints used in our original paper. We recommend using `davinci-002`, as shown below.

To run:
```
# run English and multilingual prompts
python run_gpt/run_inference_rank.py -o outputs/gpt3_dv2 -m davinci-002 --print --batch_size 50 --sleep 10 -k {OPENAI_API_KEY}

```
Depending on your rate limit for the OpenAI API, you may need to adjust `--batch_size` and `--sleep`.

#### B. Local model inference
For local models (BLOOM, T0, etc.), we use rank classification. This is implemented in `rank_outputs/`:

```
# run English and multilingual prompts
python rank_outputs/main.py -o outputs/bloomz-560m -m bigscience/bloomz-560m --batch_size 24

# run for 7b1, bloom, etc
```

#### c. GPT-4 inference
For GPT-4, we use a parsing approach. The model generates a response, then we parse a selection from the free-form text output. This allows us to perform our prompt modification experiments.

Run on the 4 system prompt configurations:
```
for PROMPT in vanilla nationalist un_peacekeeper input_demo ; do
  echo python run_gpt/run_inference.py -o outputs/gpt4/$PROMPT -m gpt-4 --system run_gpt/system_prompts/$PROMPT.txt --sleep 0
done
```

#### 2. Evaluate!
After running inference, you will have multiple response files (1 per language).
Combine them into a response table by running:

```
# run for GPT-3
python gen_response_table.py -rd outputs/gpt3_dv2

# run for BLOOMZ 560M
python gen_response_table.py -rd outputs/bloomz-560m

# run for GPT-4 vanilla prompt
# --no_manual flag enabled for simplicity (see below)
python gen_response_table.py -rd outputs/gpt4-0314/vanilla --no_manual

# modify args for outputs from other models and prompts
```

__Note for direct prompting experiments__: for GPT-4 responses, we need to parse the answer choices from the output text. The `gen_response_table.py` script will attempt to automatically parse at first. Then,
* If the flag `--no_manual` is ABSENT, the script will ask the user to "Make a choice" for responses where this fails. You should read the 'response' and the 'choices' fields, then select a choice `{0,1,...}`.
* If the flag `--no_manual` is PRESENT, the script will attempt to match the choice that first appears in the responses.
After, it will select the 0-th index.

#### 3. Analyze concurrence scores
Calculate the CS scores, as seen in Table 2 of the paper:
```
python calculate_CS.py outputs/gpt3_dv2/response_table.csv

python calculate_CS.py outputs/bloomz-560m/response_table.csv

python calculate_CS.py outputs/gpt4-0314/vanilla/response_table.csv

# modify args for outputs from other models and prompts
```

#### 4. Graph plot CSV scores
`all_results/` contains response tables and score files for all runs, after running the script `calculate_CS.sh`. That script also generates the tables, in CSV and Latex format.

See `scripts/plot_CS_results.ipynb` to generate the plots.

## Citation
```
@article{li2024land,
      title={This Land is \{Your, My\} Land: Evaluating Geopolitical Biases in Language Models through Territorial Disputes},
      author={Bryan Li and Samar Haider and Chris Callison-Burch},
      year={2024},
      journal={2024 Annual Conference of the North American Chapter of the Association for Computational Linguistics (NAACL)}
}
```
