import argparse
import json
import pandas as pd
from collections import defaultdict
import os

from lib import TERR_PATH, INFO_PATH, TEMP_PATH

LETTERS = 'ABCDEFG'

parser = argparse.ArgumentParser()
parser.add_argument('--terr_path', '-t', default=TERR_PATH)
parser.add_argument('--info_path', '-ip', default=INFO_PATH)
parser.add_argument('--translate_dir', '-td', default='translate/')
parser.add_argument('--suffix', '-su')

if __name__ == "__main__":
    args = parser.parse_args()

    df = pd.read_csv(args.terr_path)
    df['Claimants'] = df['Claimants'].str.split(';')

    with open(args.info_path, 'r') as f:
        countries_info = json.load(f)

    terms_dir = f'{args.translate_dir}/terms/'
    os.makedirs(terms_dir, exist_ok=True)

    df['Claimant_Codes'] = df['Claimants'].apply(
        lambda l: [countries_info[c]['code'] for c in l]).to_list()

    df_temp = pd.read_csv(TEMP_PATH, sep='\t')
    usable_codes = df_temp['code'].values
    skipped_codes = set()

    terms_d = defaultdict(set)
    for i, row in df.iterrows():
        for code in row['Claimant_Codes']:
            if code == 'zh':
                code = 'zh-CN'
            elif code == 'he':
                code = 'iw'
            if code not in usable_codes:
                skipped_codes.add(code)
                continue
            terms_d[code].add(row['Territory'])
            terms_d[code].update(row['Claimants'])

    for code in usable_codes:
        terms = terms_d[code]
        out_path = f'{terms_dir}/terms2trans.{code}.txt'
        print(f'Saving {len(terms)} terms to {out_path}')

        with open(out_path, 'w') as f:
            json.dump(list(terms), f)

    print(f'skipped following codes, as they are not in {TEMP_PATH}:\n {skipped_codes}')

    ### also write the terms per line
    codes_per_claimant = df['Claimants'].apply(
        lambda l: [countries_info[c]['code'] for c in l]).to_list()

    code2line = defaultdict(set)
    for i, codes_claim in enumerate(codes_per_claimant):
        for cc in codes_claim:
            if cc in skipped_codes:
                continue
            code2line[cc].add(i)

    print(f'Saving line indices to {terms_dir}/line_inds*')
    for cc, line_inds in code2line.items():
        out_path = f'{terms_dir}/line_inds.{cc}.txt'
        with open(out_path, 'w') as f:
            json.dump(sorted(line_inds), f)
