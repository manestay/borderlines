import argparse
import json
from pathlib import Path
from time import sleep

from google.oauth2.service_account import Credentials
from google.cloud import translate_v2 as translate
from googletrans import Translator

parser = argparse.ArgumentParser()
parser.add_argument('in_dir', type=Path)
parser.add_argument('--code', help='only translate to specified language')
parser.add_argument('-m', '--mt-system', choices=['gtpy', 'cloud'], default='gtpy')
parser.add_argument('--credentials', default='./gc_credentials.json')

if __name__ == "__main__":
    args = parser.parse_args()
    if args.mt_system == 'gtpy':
        client = Translator()
    elif args.mt_system == 'cloud':
        credentials = Credentials.from_service_account_file(args.credentials)
        client = translate.Client(credentials=credentials)

    for fname in args.in_dir.glob('terms2trans*'):
        code = fname.suffixes[0].lstrip('.')

        if args.code and code != args.code:
            continue

        out_path = args.in_dir / f'terms_gt.{code}.json'

        print(f'translating {fname} to {out_path}')

        with fname.open('r') as f:
            lines = json.load(f)
        if code == 'en':
            term_d = dict(zip(lines, lines))
        elif args.mt_system == 'gtpy':
            trans = client.translate('\n'.join(lines), src='en', dest=code)
            trans_text = trans.text.split('\n')
            term_d = dict(zip(lines, trans_text))
            sleep(2)
        elif args.mt_system == 'cloud':
            trans = client.translate(lines, source_language='en', target_language=code)
            term_d = {item['input']: item['translatedText'].replace('&#39;', "'") for item in trans}

        with out_path.open('w') as f:
            json.dump(term_d, f, ensure_ascii=False)
