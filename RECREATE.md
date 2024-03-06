The following commands will generate a dataset based on the [2021-08-31](https://en.wikipedia.org/w/index.php?title=List_of_territorial_disputes&oldid=1041639056) article. This date ensures the "world state" of the dataset with respect to the training data cutoff of GPT-3 (2021-09).

Or, you can pick a version from [this page](https://en.wikipedia.org/w/index.php?title=List_of_territorial_disputes&action=history), and modify the arguments accordingly.

### 1. Generate dataset
```
python get_territory_dataset.py -o disputed_territories_2021-08.csv  -ip countries_info_2021-08.json --site 2021-08-31
```

NOTE: you will likely need to perform some manual edits on the collected dataset. For example, you will want to remove the entry "Saudi Arabia–United Arab Emirates border dispute", since this is not a specific territory, but rather broadly-covered.

### 2. Generate English prompts
```
python get_prompts.py -ip countries_info_2021-08.json -t disputed_territories_2021-08.csv -s prompts_2021-08
```

### 3. Generate multilingual prompts
For this section, you need to setup [Google Cloud credentials](https://developers.google.com/workspace/guides/create-credentials), and save to `./gc_credentials.json`. This is to use Google Translate (GT)

We use a templated translation method, in which we write an English sentence for terrorital disputes, abstracting named entities with XX, YY, ZZ. For each language, we translate the sentence to it only once, and also translate the relevant entities. Then we replace the abstractions with the entities.

#### A. Translate templated sentence (skip!)
(You can **skip this step** since `templates_q.csv` is included in this repo)
Translate the templated sentence (`TEMP` in `lib.py`) from English to all supported languages using GT:
```
python translate_template.py -m cloud
```
You then need to manually ensure each line has `XX, YY, ZZ`, since GT may transliterate. For example, the row for `sr` (Serbian) may have `КСКС`, which should be changed to `XX`.

#### B. Translate entities
First, extract the entities that need to be translated for each language:
```
python get_terms_to_translate.py -t disputed_territories_2021-08.csv -ip countries_info_2021-08.json -td translate_2021-08
```

Then, run GT over the `terms/` folder:
```
python translate_folder.py translate_2021-08/terms -m cloud
```

#### C. Translate
Next, insert the entities into the templates, for each language:
```
python get_multiling_prompts.py translate_2021-08/terms translate_2021-08/prompts_mc_q -t disputed_territories_2021-08.csv
```
The version of the Borderlines dataset for your chosen article date (2023-08-31 if you used the above commands) is saved to `translate_2021-08/prompts_mc_q/prompts.{lang_code}`.

### 4. Create dataset objects
Finally, save the BorderLines dataset you created in the HuggingFace datasets format.
```
python scripts/borderlines_to_datasets_format.py -o datasets/2021-08 -p prompts_2021-08/prompts_q_mc.txt -td translate_2021-08/terms -tp disputed_territories_2021-08.csv -ip countries_info_2021-08.json -pd translate_2021-08/prompts_mc_q
```
