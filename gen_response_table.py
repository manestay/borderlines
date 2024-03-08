import argparse
import re
from pathlib import Path

from lib import LETTERS
from run_gpt.lib import load_borderlines_hf

LETTER2NUM = {x: i for i, x in enumerate(LETTERS)}

parser = argparse.ArgumentParser()
parser.add_argument('-o', '--out_path', type=Path)
parser.add_argument('--response_dir', '-rd', type=Path)
parser.add_argument('--dataset_dir', '-dd', type=Path, help='path to dataset saved locally')

parser.add_argument('--no_manual', action='store_true')

def user_input_choice(response, choices, choices_orig):
    print('response:', response)
    if choices_orig:
        print('choices:', list(enumerate(zip(choices, choices_orig))))
    else:
        print('choices:', list(enumerate(choices)))
    user_choice = input(f'Make a choice: ').strip()
    if user_choice.isdigit():
        uc = int(user_choice)
        if uc <= len(choices):
            choice = choices[uc]
            if choices_orig:
                choice_orig = choices_orig[uc]
                print(f'ADDED user choice {choice} ({choice_orig})\n')
            else:
                print(f'ADDED user choice {choice}\n')
            return uc
    print(f'ADDED first choice {choices[0]}', f'({choices_orig[0]})\n' if choices_orig else '\n')
    return 0

def parse_responses(responses, claimants, prompts, claimants_en=None, lang='', no_manual=False):
    def argmin(a):
        return a.index(min(a))

    if claimants_en is None:
        claimants_en = [None] * len(claimants)
    chosen_inds = []

    for i, (response, choices, choices_en, prompt) in enumerate(zip(responses, claimants, claimants_en, prompts), 1):
        added = False

        match_arr = [choice in response for choice in choices]
        if match_arr.count(True) == 1:
            chosen_inds.append(match_arr.index(True))
            added = True
        else:
            selected = re.findall(rf'[{LETTERS}]\)', response)
            if len(selected) == 1:
                letter = selected[0][0]
                chosen_inds.append(LETTER2NUM[letter])
                added = True

        if not added:
            lang_str = f'({lang})' if lang else ''
            print(f'could not add for row {i}{lang_str}: {response}')
            if no_manual:
                first_inds = [response.find(x) for x in choices]
                first_inds = [x if x != -1 else len(response) for x in first_inds]
                user_ind = 0 if not first_inds else argmin(first_inds)
            else:
                user_ind = user_input_choice(response, choices, choices_en)

            assert user_ind < len(choices)
            chosen_inds.append(user_ind)
            added = True

    chosen = [choices[uc] for choices, uc in zip(claimants, chosen_inds)]
    if None not in claimants_en: # 2 return
        return [choices[uc] for choices, uc in zip(claimants_en, chosen_inds)], chosen
    return chosen

if __name__ == "__main__":
    args = parser.parse_args()

    if not args.out_path:
        args.out_path = args.response_dir / 'response_table.csv'

    territories, countries, queries = load_borderlines_hf(args.dataset_dir)
    country2code = {entry['Country']: entry['Lang_Code'] for entry in countries}

    territories = territories.map(lambda row: {
        'Claimant_Codes': [country2code[c] for c in row['Claimants']],
        'Controller_Code': country2code[row['Controller']] if row['Controller'] in country2code else ''
    })
    territories = territories.to_pandas()

    responses_en_path = args.response_dir / f'responses_mc_all.txt'
    with responses_en_path.open('r') as f:
        responses_en = [x.strip() for x in f]

    picked_en = parse_responses(responses_en, territories['Claimants'], territories['Query'],
                             no_manual=args.no_manual)
    territories['Response_en'] = picked_en
    territories['Responses_d'] = [{} for x in picked_en]

    picked_non_control = [] # other claimant picked its lang
    picked_control = [] # controller picked its lang
    picked_unk = []


    for lang, ds in queries.items():
        print(f'processing {lang}    ', end='\r')

        claimants = ds['Claimants_Native']
        claimants_en = territories.iloc[ds['Index_Territory']]['Claimants']
        prompts = ds['Query_Native']

        response_path = args.response_dir / f'responses_mc.{lang}'

        if not response_path.exists():
            response_path = args.response_dir / f'responses_mc.{lang}'

        with response_path.open('r') as f:
            responses = [x.strip() for x in f.readlines()]

        picked, picked_native = parse_responses(responses, claimants, prompts, claimants_en, lang,
                                 no_manual=args.no_manual)

        if picked[0] is None and all(x == None for x in picked):
            print(f'{lang} not supported!')

        for pick, li in zip(picked, ds['Index_Territory']):
            row = territories.iloc[li]
            if pick is None:
                pass
            elif not row['Controller_Code']:
                res = country2code[pick] == lang
                picked_unk.append(res)
            # if not controller, but picks itself as lang, then True (i.e., biased!)
            elif row['Controller_Code'] != lang:
                res = country2code[pick] == row['Controller_Code']
                picked_non_control.append(res)
            else:
                res = country2code[pick] == row['Controller_Code']
                picked_control.append(res)
            territories.iloc[li]['Responses_d'][lang] = pick

    territories['Response_Controller'] = territories.apply(lambda row: row['Responses_d'].get(row['Controller_Code']), axis=1)
    territories['Unique_Claimants'] = territories['Responses_d'].apply(lambda x: set(x.values()))

    territories_save = territories.copy()
    territories_save['Claimants'] = territories_save['Claimants'].str.join(';')
    territories_save['Claimant_Codes'] = territories_save['Claimant_Codes'].str.join(';')
    territories_save.drop(['Query'], axis=1, inplace=True)

    args.out_path.parent.mkdir(exist_ok=True, parents=True)

    territories_save.to_csv(args.out_path, index=False)
    print(f'saved to {args.out_path}')

    ds = territories[territories.Unique_Claimants.apply(len) != 1]

    print('\nOverall')
    print('-' * 10)
    territories_kno = territories[territories['Controller'] != 'Unknown']
    print('known:', (territories_kno['Controller'] == territories_kno['Response_en']).mean())

    groups = territories.groupby('Region')
    for region, territoriesg in groups:
        print(region)
        print('-' * 10)
        territoriesg_kno = territoriesg[territoriesg['Controller'] != 'Unknown']
        print('known:', (territoriesg_kno['Controller'] == territoriesg_kno['Response_en']).mean())
