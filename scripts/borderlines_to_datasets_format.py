import argparse
import json
import sys
from pathlib import Path

import datasets

import pandas as pd

sys.path.append('.')

from lib import TERR_PATH, INFO_PATH, LETTERS

parser = argparse.ArgumentParser()
parser.add_argument('-o', '--out_path', type=Path)

parser.add_argument('-p', '--prompts_path', type=Path)
parser.add_argument('-pd', '--prompts_dir', type=Path)
parser.add_argument('-td', '--terms_dir', type=Path)
parser.add_argument('--terr_path', '-tp', default=TERR_PATH)
parser.add_argument('--info_path', '-ip', default=INFO_PATH)

if __name__ == "__main__":
    args = parser.parse_args()

    # make territories Dataset
    df = pd.read_csv(args.terr_path)
    df['Claimants'] = df['Claimants'].str.split(';')

    if 'Query' not in df.columns:
        with open(args.prompts_path, 'r') as f:
            prompts = [x.strip() for x in f.readlines()]
        df['Query'] = prompts

    if 'QueryID' not in df.columns:
        df['QueryID'] = df['Territory'].map(lambda x: f"{x.replace(' ', '_')}_en")

    territories = datasets.Dataset.from_pandas(df)

    # make countries Dataset
    with open(args.info_path, 'r') as f:
        countries_info = json.load(f)
    countries_l = []
    for country, d in countries_info.items():
        d['Country'] = country
        countries_l.append(d)

    countries = datasets.Dataset.from_list(countries_l)

    # make queries DatasetDict
    queries = {}
    for line_path in args.terms_dir.glob('line_inds*'):
        curr_d = {}

        code = line_path.suffixes[0].lstrip('.')
        print(f'processing {code}    ', end='\r')

        prompt_path = args.prompts_dir / f'prompts.{code}'
        with prompt_path.open('r') as f:
            prompts = [x.strip() for x in f.readlines()]

        choices_path = args.prompts_dir / f'choices.{code}'
        with choices_path.open('r') as f:
            choices_l = [x.strip() for x in f.readlines()]

        with line_path.open('r') as f:
            line_inds = json.load(f)
            assert line_inds == sorted(line_inds)

        curr_d['Query_Native'] = prompts
        curr_d['Claimants_Native'] = choices_l
        curr_d['Index_Territory'] = line_inds
        curr_d['QueryID'] = [f"{territories[x]['Territory'].replace(' ', '_')}_{code}"
                             for x in line_inds]

        curr_ds = datasets.Dataset.from_dict(curr_d)
        queries[code] = curr_ds
    queries = datasets.DatasetDict(queries)

    args.out_path.mkdir(exist_ok=True, parents=True)
    print(f'saving to {args.out_path}')
    out_path_t = args.out_path / 'territories'
    territories.save_to_disk(str(out_path_t))

    out_path_c = args.out_path / 'countries_info'
    countries.save_to_disk(str(out_path_c))

    out_path_q = args.out_path / 'queries'
    queries.save_to_disk(str(out_path_q))
