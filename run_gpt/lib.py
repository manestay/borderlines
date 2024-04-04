import os

import datasets
import openai

CHAT_MODELS = set(['gpt-4', 'gpt-3.5'])

DEFAULT_KEY_PATH = f'{os.path.expanduser("~")}/projects/openai_key.txt'
DEFAULT_ORG_PATH = f'{os.path.expanduser("~")}/projects/openai_org.txt'

LETTERS = 'ABCDEFG'
LETTERS_RE = rf'[{LETTERS}]\)'


def argmax(l):
    return l.index(max(l))


def chunks(*lists, batch_size):
    """Yield successive n-sized chunks for each list."""
    for i in range(0, len(lists[0]), batch_size):
        yield (l[i:i + batch_size] for l in lists)


def is_chat_model(model_name):
    if 'instruct' in model_name:
        return False
    return any(x in model_name for x in CHAT_MODELS)


def load_file_or_string(path):
    if not path:
        return ''

    if os.path.exists(path):
        with open(path, 'r') as f:
            s = f.read().strip()
            return s
    elif path.endswith('.txt'):
        return ''

    return path


def load_openai_client(key_path=DEFAULT_KEY_PATH, org_path=DEFAULT_ORG_PATH) -> openai.OpenAI:
    key = load_file_or_string(key_path or DEFAULT_KEY_PATH)
    org = load_file_or_string(org_path or DEFAULT_ORG_PATH)

    client = openai.OpenAI(api_key=key, organization=org)

    return client


def join_toks(tokens):
    ''' Resolves unicode bytes for echo-ed text
    See: https://community.openai.com/t/tokens-are-mangled-for-some-non-english-characters-resolved/74315/7
    '''
    byte_tokens = []
    for tok in tokens:
        if tok.startswith('bytes:'):
            char_ints = [int(char, base=16) for char in tok.split('\\x')[1:]]
            if tok.startswith('bytes: '):  # with a intermediate space:
                byte_tokens.append(' '.encode())
            byte_tokens.extend([bytes([ci]) for ci in char_ints])
        else:
            byte_tokens.append(tok.encode())
    original = b''.join(byte_tokens).decode().strip()
    return original


def langcode2lang(countries_info):
    lc2l = {}
    for entry in countries_info:
        lc2l[entry['Lang_Code']] = entry['Lang_Name']
    lc2l['zhs'] = 'Chinese'
    lc2l['zht'] = 'Traditional Chinese'
    return lc2l


def load_borderlines_hf(dataset_dir=None):
    def split_field(row, field):
        return {field: row[field].split(';')}
    split_field.__module__ = None

    if not dataset_dir:
        print('loading from the datasets hub...', end=' ')
        # load disputed territories
        territories = datasets.load_dataset('manestay/borderlines', 'territories')
        # the loaded file stores lists with ; separators, so split it
        territories = territories.map(split_field, fn_kwargs={'field': 'Claimants'})['train']

        # load country demographics
        countries = datasets.load_dataset('manestay/borderlines', 'countries')['train']

        # load queries in 49 languages
        queries = datasets.load_dataset('manestay/borderlines', 'queries')
        queries = queries.map(split_field, fn_kwargs={'field': 'Claimants_Native'})
    else:
        print(f'loading from {dataset_dir}...', end=' ')
        territories = datasets.load_from_disk(os.path.join(dataset_dir, 'territories'))
        countries = datasets.load_from_disk(os.path.join(dataset_dir, 'countries_info'))
        queries = datasets.load_from_disk(os.path.join(dataset_dir, 'queries'))
    print('done')
    return territories, countries, queries
