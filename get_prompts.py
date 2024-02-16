import argparse
import json
import os
import pandas as pd
from collections import defaultdict

from get_territory_dataset import TERR_PATH, INFO_PATH

from lib import LETTERS

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt_type', default='mc', choices=['mc', 'binary', 'mc_fewshot'])
    parser.add_argument('--terr_path', '-t', default=TERR_PATH)
    parser.add_argument('--info_path', '-ip', default=INFO_PATH)
    parser.add_argument('--save_dir', '-s', default='prompts')
    parser.add_argument('--suffix', '-su')
    return parser

def get_lang2countries(countries_info):
    lang2countries = defaultdict(list)
    for k, v in countries_info.items():
        if v:
            lang2countries[v['code']].append(k)
    return lang2countries

def get_or_phrase(choices, sep='or'):
    mc = []
    for letter, choice in zip(LETTERS, choices):
        mc.append(f'{letter}) {choice}')
    return f' {sep} '.join(mc)

def gen_mc_prompt(row):
    # prompt = f'{row["Territory"]} is a territory of {get_or_phrase(row["Claimants"])}.'
    prompt = f'Is {row["Territory"]} a territory of {get_or_phrase(row["Claimants"])}?'
    return prompt

def gen_mc_fewshot_prompt(row):
    prompt = f'''1. Alaska is a territory of A) Canada , B) USA
2. Bahia is a territory of A) Brazil , B) Argentina , C) Portugal
3. {row["Territory"]} is a territory of {get_or_phrase(row["Claimants"])}
1. B) USA
2. A) Brazil
3.'''
    return prompt.replace('\n', '\\n')

def gen_binary_prompt(row):
    prompts = []
    for claimant in row["Claimants"]:
        prompt = f'{row["Territory"]} is a territory of {claimant} (True or False)'
        prompts.append(prompt)
    return prompts

def generate_prompts(df, prompt_type):
    extra_info = None
    if prompt_type == 'mc':
        prompts = df.apply(gen_mc_prompt, axis=1).tolist()
    elif prompt_type == 'binary':
        print(f'WARNING: {prompt_type} not recommended')
        prompts_list = df.apply(gen_binary_prompt, axis=1).tolist()
        prompts = [x for subl in prompts_list for x in subl]
        extra_info = [[i] * len(subl) for i, subl in enumerate(prompts_list)]
        extra_info = [x for subl in extra_info for x in subl]
    if prompt_type == 'mc_fewshot':
        print(f'WARNING: {prompt_type} not recommended')
        prompts = df.apply(gen_mc_fewshot_prompt, axis=1).tolist()
    return prompts, extra_info

if __name__ == "__main__":
    args = get_parser().parse_args()

    df = pd.read_csv(args.terr_path)
    df['Claimants'] = df['Claimants'].str.split(';')

    with open(args.info_path, 'r') as f:
        countries_info = json.load(f)

    prompts, extra_info = generate_prompts(df, args.prompt_type)

    os.makedirs(args.save_dir, exist_ok=True)
    save_path = os.path.join(args.save_dir, f'prompts_q_{args.prompt_type}.txt')
    print(f'saving {len(prompts)} prompts to {save_path}')

    with open(save_path, 'w') as f:
        for line in prompts:
            f.write(line + '\n')

    if extra_info:
        extra_path = os.path.join(args.save_dir, f'extra.txt')
        print(f'saving extra info to {extra_path}')

        with open(extra_path, 'w') as f:
            for line in extra_info:
                f.write(str(line) + '\n')

    choices = df['Claimants'].apply(json.dumps, ensure_ascii=False).to_list()
    with open(os.path.join(args.save_dir, 'choices.txt'), 'w') as f:
        f.writelines(line + '\n' for line in choices)
