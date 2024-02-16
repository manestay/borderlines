import argparse
from copy import copy
import json
from pathlib import Path
from time import sleep

from lib import load_openai_client, load_file_or_string, is_chat_model, langcode2lang

INFO_PATH = 'countries_info.json'
SEED = 2557
LETTERS = 'ABCDEFG'
LETTERS_RE = f'[{LETTERS}]\)'

client = None

parser = argparse.ArgumentParser()
parser.add_argument('--api-key', '-k')
parser.add_argument('--org', '-or')
parser.add_argument('--in_path', '-i', default=None, type=Path)
parser.add_argument('--out_path', '-o', default=None, type=Path)
parser.add_argument('--write_mode', '-wm', default='w', choices=['w','a'])
parser.add_argument('--batch_size', '-b', type=int, default=1)
parser.add_argument('--sleep', type=float, default=1)
parser.add_argument('--model', '-m', default="text-davinci-003")
parser.add_argument('--try_fix', '-f', action='store_true')
parser.add_argument('--max_tokens', '-tok', type=int, default=128)
parser.add_argument('--system_prompt', '-sys', default='run_gpt/system_prompts/default.txt')
parser.add_argument('--terms_path', '-tm', type=Path)
parser.add_argument('--info_path', '-ip', default=INFO_PATH)

def add_demographics(system_prompt, choices, countries_info):
    arr = []
    for country in choices:
        religion = countries_info[country]['religion']
        lang = countries_info[country]['name']

        arr.append(f'{country}, Religion: {religion}, Language: {lang}')


    replace_str = '\n'.join(arr)
    system_prompt = system_prompt.replace('|demo|', replace_str)

    return system_prompt

def add_nationalist(system_prompt, choices, countries_info, prompt_lc):
    if prompt_lc == 'txt':
        replace_str = f' Begin your response with "As a current resident,"'
    else:
        replace_str = ''
        for country in choices:
            lang = countries_info[country]['name']
            lang_code = countries_info[country]['code']
            if prompt_lc == lang_code:
                replace_str = f' Begin your response with "As a citizen of {country},"'
                break
        if not replace_str:
            import pdb; pdb.set_trace()
        if prompt_lc != 'en':
            replace_str += f' (translated to {lang})'
    system_prompt = system_prompt.replace('|nationalist|', replace_str)
    return system_prompt

def get_completion(prompt, model, max_tokens, system_prompt, choices=[], lc=None):
    global client
    sp = ''
    if is_chat_model(model):
        if len(prompt) > 1:
            print('WARNING: ChatCompletions does not support batching, please change batch size to 1')
            exit(-1)

        messages = []
        if system_prompt:
            if '|demo|' in system_prompt:
                system_prompt = add_demographics(system_prompt, choices, countries_info)
            if '|nationalist|' in system_prompt:
                system_prompt = add_nationalist(system_prompt, choices, countries_info, lc)
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
            print('WARNING: passed in `system_prompt`, but this is not used with Completions')
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

def is_valid(response):
    return response.count(')') == 1

let2num = {f'{l})': f'{n})' for n, l in enumerate('ABCDEFG', 1)}

def try_variants(text, model):
    '''
    Attempt prompt variants, else ask user for a manual fix
    '''
    text_ = text[:-2] + '?\n'
    responses1, sp = get_completion([text_], model)
    if is_valid(responses1[0]):
        return responses1

    text_ = text
    for k, v in let2num.items():
        text_ = text_.replace(k, v, 1)
    responses2, sp = get_completion([text_], model)
    for k, v in let2num.items():
        responses2[0] = responses2[0].replace(v, k, 1)
    if is_valid(responses2[0]):
        return responses2

    text_ = text[:-1] + ' Pick only one.\n'
    responses3, sp = get_completion([text_], model)
    if is_valid(responses3[0]):
        return responses3

    text_ = 'Q: ' + text[:-2] + '?\nA:'
    responses4, sp = get_completion([text_], model)
    if is_valid(responses4[0]):
        return responses4

    l = [responses1, responses2, responses3, responses4]
    print('Prompt -', text)
    print('0 -', responses1)
    print('1 -', responses2)
    print('2 -', responses3)
    print('3 -', responses4)
    user_choice = input('Pick one number from above (or enter own string). ')
    if user_choice.isnumeric():
        responses_ = l[int(user_choice)]
    else:
        responses_ = [user_choice]
    return responses_


if __name__ == "__main__":
    args = parser.parse_args()
    client = load_openai_client(args.api_key, args.org)

    system_prompt = ''

    if args.in_path:
        assert args.out_path, "must use -i with -o"
        write_mode = args.write_mode

        if write_mode == 'a':
            # assert args.batch_size == 1, "cannot use 'a' write mode with batch size"
            wc_out = sum(1 for _ in args.out_path.open('r'))
        elif write_mode == 'w':
            args.out_path.parent.mkdir(exist_ok=True, parents=True)

        if args.try_fix:
            print('setting batch size to 1 for --try_fix')
            args.batch_size = 1

        choices_l = None
        if is_chat_model(args.model):
            print('using batch size of 1 for ChatCompletion')
            args.batch_size = 1

            system_prompt = load_file_or_string(args.system_prompt)
            if args.system_prompt and not system_prompt:
                print(f'WARNING: could not load {args.system_prompt}')

            if '|' in system_prompt:
                with open(args.info_path, 'r') as f:
                    countries_info = json.load(f)
                code = args.in_path.suffixes[0][1:]
                choices_path = args.in_path.parent / f'choices.{code}'
                with open(choices_path, 'r') as f:
                    choices_l = [json.loads(x) for x in f]
                if args.terms_path:
                    terms_path = args.terms_path / f'terms_gt.{code}.json'
                    with open(terms_path, 'r') as f:
                        terms_d = json.load(f)
                        terms_d = {v: k for k, v in terms_d.items()}
                    choices_l_ = []
                    for choices in choices_l:
                        choices_l_.append([terms_d[choice] for choice in choices])
                    choices_l = choices_l_

            lc = args.in_path.suffix[1:]

            if '|lang|' in system_prompt:
                lc2l = langcode2lang(countries_info)

                if lc == 'en' or lc == 'txt':
                    replace_str = ''
                elif lc not in lc2l:
                    print(f'WARNING: could not find code for {lc}')
                else:
                    replace_str = f' (translated to {lc2l[lc]})'

                system_prompt = system_prompt.replace('|lang|', replace_str)

            print('using system prompt:', system_prompt)

        f_op = None
        if '|' in system_prompt:
            out_prompt_path =  args.out_path.parent / f'prompts_used{args.out_path.suffix}'
            f_op = out_prompt_path.open('w')

        batch = []
        with args.in_path.open('r') as f_in, args.out_path.open(write_mode) as f_out:
            for i, line in enumerate(f_in):
                if write_mode == 'a' and i <= wc_out:
                        continue
                batch.append(line)

                choices = choices_l[i] if choices_l is not None else None # only works for batch size 1
                if len(batch) == args.batch_size:
                    responses, sp = get_completion(batch, args.model, args.max_tokens, system_prompt, choices, lc)
                    if args.try_fix:
                        if not is_valid(responses[0]):
                            responses = try_variants(batch[0], args.model)
                    f_out.writelines([response + '\n' for response in responses])
                    if f_op: # only works for batch size 1
                        f_op.write(sp + '\n')

                    print(f'processed line {i}; |Q| {batch[-1].strip()} ; |A| {responses[-1]}' +
                          ' ' * 20, end='\r')
                    batch = []
                sleep(args.sleep)
            if batch:
                choices = choices_l[i]
                responses, sp = get_completion(batch, args.model, args.max_tokens, system_prompt, choices, lc)
                f_out.writelines([response + '\n' for response in responses])
                if f_op: # only works for batch size 1
                    f_op.write(sp + '\n')
        print('done')
    else:
        while True:
            article = input('Enter text:\n')
            if not article: continue
            print('-----')
            qa_pairs, sp = get_completion(article, args.model, args.max_tokens, system_prompt)
            print(qa_pairs)
