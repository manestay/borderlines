'''
Translate the multiple-choice prompt template to target languages
NOTE: we perform some post-processing to manually fix after!
'''

import argparse
from time import sleep

from google.oauth2.service_account import Credentials
from google.cloud import translate_v2 as translate
from googletrans import Translator

from lib import TEMP_PATH, CODES, TEMPLATE

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--mt-system', choices=['gtpy', 'cloud'], default='gtpy')
parser.add_argument('--credentials', default='./gc_credentials.json')

if __name__ == "__main__":
    args = parser.parse_args()
    if args.mt_system == 'gtpy':
        client = Translator()
    elif args.mt_system == 'cloud':
        credentials = Credentials.from_service_account_file(args.credentials)
        client = translate.Client(credentials=credentials)

    templates = []
    with open(TEMP_PATH, 'w') as f:
        f.write('code\ttemplate\n')
        for code in CODES:
            print(f'processing {code}', end='')
            trans = None
            if code == 'en':
                trans_text = TEMPLATE
            elif args.mt_system == 'gtpy':
                while not trans:
                    try:
                        trans = client.translate(TEMPLATE, src='en', dest=code)
                        trans_text = trans.text
                    except ValueError:
                        print(' - invalid code, skipped')
                        break
                    except TypeError:
                        print(' - failed, trying again in 5s')
                        sleep(5)
                sleep(1.5)
            elif args.mt_system == 'cloud':
                trans = client.translate(TEMPLATE, source_language='en', target_language=code)
                trans_text = trans["translatedText"]
            f.write(f'{code}\t{trans_text}\n')
            print('\n', trans_text)
