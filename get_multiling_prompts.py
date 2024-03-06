import argparse
import json
import re
from pathlib import Path

import pandas as pd

from get_prompts import get_or_phrase

from lib import TERR_PATH, TEMP_PATH

parser = argparse.ArgumentParser()
parser.add_argument('in_dir', type=Path)
parser.add_argument('out_dir', type=Path)
parser.add_argument('--terr_path', '-t', default=TERR_PATH)

def get_sep(s):
    return re.search('(?<=YY).*(?=ZZ)', s)[0].strip()

def get_prompt(template, territory, claimants):
    template = template.replace('XX', territory, 1)
    sep = get_sep(template)
    template = re.sub(r'YY.*ZZ', get_or_phrase(claimants, sep), template, 1)
    return template


if __name__ == "__main__":
    args = parser.parse_args()

    args.out_dir.mkdir(exist_ok=True, parents=True)

    df = pd.read_csv(args.terr_path)
    df['Claimants'] = df['Claimants'].str.split(';')

    templates = pd.read_csv(TEMP_PATH, sep='\t').set_index('code')
    template_d = templates['template'].to_dict()

    for fname in args.in_dir.glob('terms_gt*'):
        code = fname.suffixes[0].lstrip('.')

        code_orig = code

        fname_line = fname.parent / f'line_inds.{code}.txt'
        out_path = args.out_dir / f'prompts.{code}'
        out_choices_path = args.out_dir / f'choices.{code}'

        template = template_d[code_orig]

        with fname.open('r') as f:
            term_d = json.load(f)

        with fname_line.open('r') as f:
            line_inds = json.load(f)

        print(f'writing {len(line_inds)} to {out_path} , {out_choices_path}')

        with out_path.open('w') as f_out, out_choices_path.open('w') as f_outc:
            for li in line_inds:
                row = df.iloc[li]
                claimants = [term_d[x] for x in row['Claimants']]
                prompt = get_prompt(template, term_d[row['Territory']], claimants)
                f_out.write(prompt + '\n')
                f_outc.write(json.dumps(claimants,ensure_ascii=False) + '\n')
