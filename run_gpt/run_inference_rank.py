import argparse
from copy import copy
import json
from pathlib import Path
import re
from time import sleep

import openai
import tiktoken
import os

from tqdm import tqdm

from lib import load_api_key, join_toks, argmax, is_chat_model

LETTERS = 'ABCDEFG'
MC_REGEX = rf'[{LETTERS}]\)'
encoding = tiktoken.get_encoding('gpt2')
client = None

parser = argparse.ArgumentParser()
parser.add_argument('--api-key', '-k')
parser.add_argument('--in_path', '-i', required=True, type=Path)
parser.add_argument('--out_path', '-o', required=True, type=Path)
parser.add_argument('--choices_path', '-c', required=False)
parser.add_argument('--write_mode', '-wm', default='w', choices=['w','a'])
parser.add_argument('--model', '-m', default="text-davinci-003")
parser.add_argument('--batch_size', '-b', type=int, default=1)
parser.add_argument('--sleep', type=float, default=1)
parser.add_argument('--print_sample', action='store_true')


def num_tokens_from_string(string: str, add_space: bool=True) -> int:
    """Returns the number of tokens in a text string."""
    string = string if not add_space else ' ' + string
    num_tokens = len(encoding.encode(string))
    return num_tokens


def get_probs(prompt, choices, model):
    global client
    if not client:
        client = openai.OpenAI()
    if is_chat_model(model):
        messages = [
            {'role': 'system', 'content': 'You are a geopolitical expert. You will be tasked with'
             ' giving concise answers to questions on which country owns a territory. Please always'
             ' select an answer from given options, and avoid saying unknown. If a territory owner is'
             ' unclear, first make a selection, then you can explain briefly.'},
             {'role': 'user', 'content': prompt[0]}
        ]
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.0,
            top_p=1,
            max_tokens=10)
        import pdb; pdb.set_trace()
        # TODO: update when logprobs released for chatcompletions
    else:
        response = client.completions.create(
            model=model,
            prompt=prompt,
            temperature=0.0,
            top_p=1,
            max_tokens=0,
            echo=True,
            logprobs=0)

    token_logprobs =  [rc.logprobs.token_logprobs for rc in response.choices]
    tokens =  [rc.logprobs.tokens for rc in response.choices]
    num_toks_per_choice = [num_tokens_from_string(choice) for choice in choices]

    prob_choices = [tl[-nt:] for tl, nt in zip(token_logprobs, num_toks_per_choice)]
    choices_echo = [join_toks(tt[-nt:]) for tt, nt in zip(tokens, num_toks_per_choice)]
    # print(choices_echo)
    assert choices == choices_echo

    cumprob_choices =  [sum(pc) for pc in prob_choices]

    return cumprob_choices

def get_choices_batch(choices_batch, probs, num_choices_q, prompts=None):
    choice_offset = 0
    choice_per_prompt = []
    printed = False
    for ncp in num_choices_q:
        choice_probs = probs[choice_offset:choice_offset+ncp]
        q_choices = choices_batch[choice_offset:choice_offset+ncp]
        pred_ind = argmax(choice_probs)
        if all([re.match(MC_REGEX, x) for x in q_choices]):
            choice_per_prompt.append(q_choices[pred_ind])
        else:
            choice_per_prompt.append(f'{LETTERS[pred_ind]}) {q_choices[pred_ind]}')
        choice_offset += ncp
        if prompts and not printed:
            printed = True
            print(prompts[0])
            print(list(zip(q_choices, choice_probs)), f'Chosen: {choice_per_prompt[0]}')

    return choice_per_prompt

if __name__ == "__main__":
    args = parser.parse_args()
    key = load_api_key(args.api_key)

    write_mode = args.write_mode

    if write_mode == 'a':
        print(f'WARNING: `append` mode is not fully tested!')
        # assert args.batch_size == 1, "cannot use 'a' write mode with batch size"
        wc_out = sum(1 for _ in args.out_path.open('r'))
    wc_in = sum(1 for _ in args.in_path.open('r'))

    # if any(x in args.model for x in CHAT_MODELS):
    #     print('setting batch size to 1 for --try_fix')
    #     args.batch_size = 1

    batch = []
    choices_batch = []
    num_choices_q = []

    pbar = tqdm(total=wc_in)
    if args.choices_path:
        f_choices = Path(args.choices_path).open('r')
    else:
        f_choices = [None] * wc_in
        print(f'--choices_path not specified, using letter choices A), B), C) ...')

    args.out_path.parent.mkdir(exist_ok=True, parents=True)

    with args.in_path.open('r') as f_in, args.out_path.open(write_mode) as f_out:
        for i, (line, line_choices) in enumerate(zip(f_in, f_choices)):
            line = line.strip()
            if write_mode == 'a' and i <= wc_out:
                continue
            if args.choices_path:
                choices = json.loads(line_choices)
            else:
                choices = re.findall(MC_REGEX, line)

            prompts_with_choices = [f'{line} Answer: {choice}' for choice in choices]

            batch_size = max(args.batch_size, len(prompts_with_choices))
            if len(batch) + len(prompts_with_choices) <= batch_size:
                batch.extend(prompts_with_choices)
                choices_batch.extend(choices)
                num_choices_q.append(len(prompts_with_choices))
                continue

            probs = get_probs(batch, choices_batch, args.model)
            batch_print = batch if args.print_sample else None
            choice_per_prompt = get_choices_batch(choices_batch, probs, num_choices_q, batch_print)
            f_out.writelines([choice + '\n' for choice in choice_per_prompt])

            pbar.update(len(num_choices_q))
            print(f'processed line {i}')

            batch = prompts_with_choices
            choices_batch = choices
            num_choices_q = [len(prompts_with_choices)]
            sleep(args.sleep)
        # TODO: process last batch
        if batch:
            probs = get_probs(batch, choices_batch, args.model)
            batch_print = batch if args.print_sample else None
            choice_per_prompt = get_choices_batch(choices_batch, probs, num_choices_q, batch_print)
            f_out.writelines([choice + '\n' for choice in choice_per_prompt])

    if args.choices_path:
        f_choices.close()

    print('done')
    pbar.close()
