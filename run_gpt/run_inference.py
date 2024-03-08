import argparse
import json
from copy import copy
from pathlib import Path
from time import sleep

from tqdm import tqdm

from lib import (is_chat_model, langcode2lang, load_borderlines_hf, chunks,
                 load_file_or_string, load_openai_client, LETTERS)

SEED = 2557
client = None

parser = argparse.ArgumentParser()
parser.add_argument('--api-key', '-k')
parser.add_argument('--org', '-or')
parser.add_argument('--dataset_dir', '-dd', type=Path, help='path to dataset saved locally')
parser.add_argument('--out_path', '-o', default=None, type=Path)
parser.add_argument('--batch_size', '-b', type=int, default=1)
parser.add_argument('--sleep', type=float, default=1)
parser.add_argument('--model', '-m', default="text-davinci-003")
parser.add_argument('--try_fix', '-f', action='store_true')
parser.add_argument('--max_tokens', '-tok', type=int, default=128)
parser.add_argument('--system_prompt', '-sys', default='run_gpt/system_prompts/vanilla.txt')


def add_demographics(system_prompt, choices, countries_info):
    arr = []
    for country in choices:
        religion = countries_info[country]['religion']
        lang = countries_info[country]['name']

        arr.append(f'{country}, Religion: {religion}, Language: {lang}')

    replace_str = '\n'.join(arr)
    system_prompt = system_prompt.replace('|demo|', replace_str)

    return system_prompt


def add_nationalist(system_prompt, choices, countries_info, prompt_lang):
    if prompt_lang == 'txt':
        replace_str = f' Begin your response with "As a current resident,".'
    else:
        replace_str = ''
        for country in choices:
            lang = countries_info[country]['Lang_Name']
            lang_code = countries_info[country]['Lang_Code']
            if prompt_lang == lang_code:
                replace_str = f' Begin your response with "As a citizen of {country},"'
                break
        if prompt_lang == 'en':
            replace_str += '.'
        else:
            replace_str += f' (translated to {lang}).'
    system_prompt = system_prompt.replace('|nationalist|', replace_str)
    return system_prompt


def get_completion(prompt, model, max_tokens, system_prompt, choices=[], lang=None, countries_d=None):
    global client


    sp = ''
    if is_chat_model(model):
        if len(prompt) > 1:
            print('WARNING: ChatCompletions does not support batching, please change batch size to 1')
            exit(-1)

        choices = choices[0]

        messages = []
        if system_prompt:
            if '|demo|' in system_prompt:
                system_prompt = add_demographics(system_prompt, choices, countries_d)
            if '|nationalist|' in system_prompt:
                system_prompt = add_nationalist(system_prompt, choices, countries_d, lang)
            messages.append({'role': 'system', 'content': system_prompt})
            sp = system_prompt.replace('\n', '\\n')
        messages.append({'role': 'user', 'content': prompt[0]})
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.0,
            top_p=1,
            seed=SEED,
            max_tokens=max_tokens)
        completions = [x.message.content for x in response.choices]
        completions = [x.replace('\n', '\\n') for x in completions]
    else:
        if system_prompt:
            print('WARNING: for a non-chat model, you should likely use run_inference_rank.py')
        response = client.completions.create(
            model=model,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=0.0,
            top_p=1,
            frequency_penalty=0.0,
            presence_penalty=0.0)
        completions = [x.text.strip() for x in response.choices]
        completions = [x.replace('\n', '\\n') for x in completions]
    return completions, sp


def add_lang_phrase(system_prompt, lang, countries):
    if '|lang|' not in system_prompt:
        return system_prompt

    lang2l = langcode2lang(countries)
    if lang == 'en' or lang == 'txt':
        replace_str = ''
    elif lang not in lang2l:
        print(f'WARNING: could not find code for {lang}')
        replace_str = ''
    else:
        replace_str = f' (translated to {lang2l[lang]})'

    system_prompt = system_prompt.replace('|lang|', replace_str)
    return system_prompt


def run_inference(query_l, choices_l, system_prompt, out_name, lang, countries, args):
    system_prompt = add_lang_phrase(system_prompt, lang, countries)
    if system_prompt:
        print('using system prompt:', system_prompt)

    countries_d = {entry['Country']: entry for entry in countries}

    f_op = None
    if '|' in system_prompt:
        # if dynamic prompt, save prompts for debugging
        out_prompt_path = out_name / f'prompts_used.{lang}'
        f_op = out_prompt_path.open('w')

    with out_name.open('w') as f_out, tqdm(total=len(query_l)) as pbar:
        for i, (batch, choices_batch) in enumerate(chunks(query_l, choices_l, batch_size=args.batch_size)):
            responses, sp = get_completion(
                batch, args.model, args.max_tokens, system_prompt, choices_batch, lang, countries_d)

            f_out.writelines([response + '\n' for response in responses])
            if f_op:
                f_op.write(sp + '\n')

            pbar.update(len(batch))
            if i % 10 == 0:
                print(f'processed line {i}\n|Q| {batch[-1].strip()}\n|A| {responses[-1]}' +
                    ' ' * 20, end='\r')
            batch = []
            choices_batch = []
            sleep(args.sleep)
    if f_op:
        f_op.close()


if __name__ == "__main__":
    args = parser.parse_args()

    client = load_openai_client(args.api_key, args.org)

    args.out_path.mkdir(exist_ok=True, parents=True)

    territories, countries, queries = load_borderlines_hf(args.dataset_dir)

    out_all = args.out_path / 'responses_mc_all.txt'
    query_l = territories['Query']
    choices_l = territories['Claimants']

    system_prompt = ''

    # if chat model, build the system prompt
    choices_l = None
    if is_chat_model(args.model):
        print('using batch size of 1 for ChatCompletion')
        args.batch_size = 1

        system_prompt = load_file_or_string(args.system_prompt)
        if args.system_prompt and not system_prompt:
            print(f'WARNING: could not load {args.system_prompt}')

    # process English
    query_l = territories['Query']
    choices_l = territories['Claimants']
    out_all = args.out_path / 'responses_mc_all.txt'
    lang = 'txt'

    run_inference(query_l, choices_l, system_prompt, out_all, lang, countries, args)

    # process multilingual
    # larger max tokens for non-Latin tokenization
    args.max_tokens = max(512, args.max_tokens)
    for lang, ds in queries.items():
        query_l = ds['Query_Native']
        # choices_l = ds['Claimants_Native']
        # use the English name of each claimant
        choices_l = territories[ds['Index_Territory']]['Claimants']
        out_path = args.out_path / f'responses_mc.{lang}'

        run_inference(query_l, choices_l, system_prompt, out_path, lang, countries, args)

    print('done')
