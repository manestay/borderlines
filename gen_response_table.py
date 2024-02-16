import argparse
import json
import re
from pathlib import Path

import pandas as pd

from lib import TERR_PATH, INFO_PATH, CODES, LETTERS

LETTER2NUM = {x: i for i, x in enumerate(LETTERS)}

parser = argparse.ArgumentParser()
parser.add_argument('terms_dir', type=Path)
parser.add_argument('response_dir', type=Path)
parser.add_argument('out_path', type=Path)
parser.add_argument('prompt_dir', type=Path, nargs='?')
parser.add_argument('--terr_path', '-t', default=TERR_PATH)
parser.add_argument('--info_path', '-ip', default=INFO_PATH)
parser.add_argument('--no_manual', action='store_true')

def user_input_choice(prompt, response, choices, choices_orig):
    if prompt:
        print('prompt:', prompt)
    print('response:', response)
    print('choices:', list(enumerate(zip(choices, choices_orig))))
    user_choice = input(f'Make a choice: ').strip()
    if user_choice.isdigit():
        uc = int(user_choice)
        if uc <= len(choices):
            choice = choices[uc]
            choice_orig = choices_orig[uc]

            print(f'ADDED user choice {choice} ({choice_orig})')
            return choice
    print(f'ADDED first choice {choices[0]} ({choices_orig[0]})')
    return choices[0]

def parse_responses(responses, claimants, prompts=[], term_translations=None, code='', no_manual=False):
    if prompts == []:
        prompts = [None] * len(claimants)
    results = []
    for i, (response, choices, prompt) in enumerate(zip(responses, claimants, prompts), 1):
        # catch case where language is unsupported (indicated by a manual fix)
        if response == 'Unsupported!':
            results = [None] * len(responses)
            return results

        choices_orig = choices
        if term_translations:
            choices = [term_translations[x] for x in choices]
        added = False

        match_arr = [choice in response for choice in choices]
        if match_arr.count(True) == 1:
            results.append(choices[match_arr.index(True)])
            added = True
        else:
            selected = re.findall(f'[{LETTERS}]\)', response)
            if len(selected) == 1:
                letter = selected[0][0]
                results.append(choices[LETTER2NUM[letter]])
                added = True

            elif len(selected) > 1:
                print(f'multiple ) for line {i} ({code}):')

        if not added:
            print(f'could not add for row {i} ({code}):')
            if no_manual:
                choice = choices[0]
            else:
                choice = user_input_choice(prompt, response, choices, choices_orig)
            if choice:
                assert choice in choices
                results.append(choice)
                added = True

    if term_translations:
        translations_rev = {v: k for k,v in term_translations.items()}
        results = [translations_rev[x] for x in results]

    return results

if __name__ == "__main__":
    args = parser.parse_args()

    if '_' in args.prompt_dir.parent.name:
        print('WARNING: you have specified a custom verison of prompts, so you will likely need to'
              ' change the --terr_path and --info_path args.')

    df = pd.read_csv(args.terr_path)
    df['Claimants'] = df['Claimants'].str.split(';')

    with open(args.info_path, 'r') as f:
        countries_info = json.load(f)

    df['Claimant_Codes'] = df['Claimants'].apply(
        lambda l: [countries_info[c]['code'] for c in l]).to_list()
    df['Controller_Code'] = df['Controller'].apply(
        lambda c: countries_info[c]['code'] if c in countries_info else '').to_list()
    # English answers
    responses_en_path = args.response_dir / f'responses_mc_all.txt'
    with responses_en_path.open('r') as f:
        responses_en = [x.strip() for x in f]
    picked = parse_responses(responses_en, df['Claimants'], no_manual=args.no_manual)
    # picked = df['Claimant_Codes']

    df['Response_en'] = picked
    # df['Responses_d'] = [{'en': x} for x in picked]
    df['Responses_d'] = [{} for x in picked]

    picked_non_control = [] # other claimant picked its lang
    picked_control = [] # controller picked its lang
    picked_unk = []

    # process answers per lang
    for line_path in args.terms_dir.glob('line_inds*'):
        code = line_path.suffixes[0].lstrip('.')
        print(f'processing {code}    ', end='\r')
        # if code != 'he':
        #     continue


        response_path = args.response_dir / f'responses_mc.{code}'

        if not response_path.exists():
            # HACK: 2 google-translate lang codes are inconsistent with ISO.
            # we manually account for it here, but should probably fix upstream.
            if code == 'zh-CN':
                code = 'zh'
            if code == 'zh':
                code = 'zh-CN'
            if code == 'iw':
                code = 'he'
            if code == 'he':
                code = 'iw'
            response_path = args.response_dir / f'responses_mc.{code}'

        with response_path.open('r') as f:
            responses = [x.strip() for x in f.readlines()]

        if args.prompt_dir:
            prompt_path = args.prompt_dir / f'prompts.{code}'
            with prompt_path.open('r') as f:
                prompts = [x.strip() for x in f.readlines()]
        else:
            prompts = []

        with line_path.open('r') as f:
            line_inds = json.load(f)
            assert line_inds == sorted(line_inds)

            if len(responses) != len(line_inds):
                print(f'warning: expected {len(line_inds)} lines, but have {len(responses)}')

        term_translations = None
        translate_path = args.terms_dir / f'terms_gt.{code}.json'
        if translate_path.exists():
            with translate_path.open('r') as f:
                term_translations = json.load(f)

        picked = parse_responses(responses, df.iloc[line_inds]['Claimants'], prompts,
                                 term_translations, code, no_manual=args.no_manual)

        if picked[0] is None and all(picked[0] == x for x in picked):
            print(f'{code} not supported!')

        for pick, li in zip(picked, line_inds):
            row = df.loc[li]
            if pick is None:
                pass
            elif not row['Controller_Code']:
                res = countries_info[pick]['code'] == code
                picked_unk.append(res)
            # if not controller, but picks itself as lang, then True (i.e., biased!)
            elif row['Controller_Code'] != code:
                res = countries_info[pick]['code'] == row['Controller_Code']
                picked_non_control.append(res)
            else:
                res = countries_info[pick]['code'] == row['Controller_Code']
                picked_control.append(res)
            df.loc[li]['Responses_d'][code] = pick

    df['Response_Controller'] = df.apply(lambda row: row['Responses_d'].get(row['Controller_Code']), axis=1)
    df['Unique_Claimants'] = df['Responses_d'].apply(lambda x: set(x.values()))

    df_save = df.copy()
    df_save['Claimants'] = df_save['Claimants'].str.join(';')
    df_save['Claimant_Codes'] = df_save['Claimant_Codes'].str.join(';')
    # df_save.drop(['Controller_Code', 'Claimant_Codes'], axis=1, inplace=True)
    df_save.to_csv(args.out_path, index=False)

    ds = df[df.Unique_Claimants.apply(len) != 1]

    print('\nOverall')
    print('-' * 10)
    df_kno = df[df['Controller'] != 'Unknown']
    print('known:', (df_kno['Controller'] == df_kno['Response_en']).mean())

    groups = df.groupby('Region')
    for region, dfg in groups:
        print(region)
        print('-' * 10)
        dfg_kno = dfg[dfg['Controller'] != 'Unknown']
        print('known:', (dfg_kno['Controller'] == dfg_kno['Response_en']).mean())
