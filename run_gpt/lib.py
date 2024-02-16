import openai
import os

CHAT_MODELS = set(['gpt-4', 'gpt-3.5'])

DEFAULT_KEY_PATH = f'{os.path.expanduser("~")}/projects/openai_key.txt'
DEFAULT_ORG_PATH = f'{os.path.expanduser("~")}/projects/openai_org.txt'

def argmax(l):
    return l.index(max(l))

def is_chat_model(model):
    return any(x in model for x in CHAT_MODELS)

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
            if tok.startswith('bytes: '): # with a intermediate space:
                byte_tokens.append(' '.encode())
            byte_tokens.extend([bytes([ci]) for ci in char_ints])
        else:
            byte_tokens.append(tok.encode())
    original = b''.join(byte_tokens).decode().strip()
    return original


def langcode2lang(countries_info):
    lc2l = {}
    for v in countries_info.values():
        lc2l[v['code']] = v['name']
    lc2l['zh-CN'] = 'Chinese'
    lc2l['zh-TW'] = 'Traditional Chinese'
    lc2l['iw'] = 'Hebrew'
    return lc2l
