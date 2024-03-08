import argparse
import json
import re
from pathlib import Path
from time import sleep

import openai
import tiktoken
from tqdm import tqdm

from lib import argmax, is_chat_model, join_toks, load_openai_client, load_borderlines_hf, LETTERS, LETTERS_RE

encoding = tiktoken.get_encoding('gpt2')
client = None

parser = argparse.ArgumentParser()
parser.add_argument('--api-key', '-k')
parser.add_argument('--org', '-or')
parser.add_argument('--dataset_dir', '-dd', type=Path, help='path to dataset saved locally')
parser.add_argument('--out_path', '-o', required=True, type=Path)
parser.add_argument('--model', '-m', default="text-davinci-003")
parser.add_argument('--batch_size', '-b', type=int, default=1)
parser.add_argument('--sleep', type=float, default=1)
parser.add_argument('--print_sample', action='store_true')


def num_tokens_from_string(string: str, add_space: bool = True) -> int:
    """Returns the number of tokens in a text string."""
    string = string if not add_space else ' ' + string
    num_tokens = len(encoding.encode(string))
    return num_tokens

def rindex(l, val):
    for index, item in enumerate(reversed(l)):
        if item == val:
            return len(l) - index - 1
            break



def get_probs(prompt, choices, model_name):
    global client
    if is_chat_model(model_name):
        messages = [
            {'role': 'system', 'content': 'You are a geopolitical expert. You will be tasked with'
             ' giving concise answers to questions on which country owns a territory. Please always'
             ' select an answer from given options, and avoid saying unknown. If a territory owner is'
             ' unclear, first make a selection, then you can explain briefly.'},
            {'role': 'user', 'content': prompt[0]}
        ]
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0.0,
            top_p=1,
            max_tokens=10)
        import pdb
        pdb.set_trace()
        # TODO: update when logprobs released for chatcompletions
    else:
        response = client.completions.create(
            model=model_name,
            prompt=prompt,
            temperature=0.0,
            top_p=1,
            max_tokens=0,
            echo=True,
            logprobs=0)

    token_logprobs = [rc.logprobs.token_logprobs for rc in response.choices]
    tokens = [rc.logprobs.tokens for rc in response.choices]
    colon_index = [rindex(x,  ':') + 1 for x in tokens]

    prob_choices = [tl[idx:] for tl, idx in zip(token_logprobs, colon_index)]
    choices_echo = [join_toks(tt[idx:]) for tt, idx in zip(tokens, colon_index)]

    assert choices == choices_echo

    cumprob_choices = [sum(pc) for pc in prob_choices]

    return cumprob_choices


def get_choices_batch(choices_batch, probs, num_choices_q, prompts=None):
    choice_offset = 0
    choice_per_prompt = []
    printed = False
    for ncp in num_choices_q:
        choice_probs = probs[choice_offset:choice_offset+ncp]
        q_choices = choices_batch[choice_offset:choice_offset+ncp]
        pred_ind = argmax(choice_probs)
        if all([re.match(LETTERS_RE, x) for x in q_choices]):
            choice_per_prompt.append(q_choices[pred_ind])
        else:
            choice_per_prompt.append(f'{LETTERS[pred_ind]}) {q_choices[pred_ind]}')
        choice_offset += ncp
        if prompts and not printed:
            printed = True
            print(prompts[0])
            print(list(zip(q_choices, choice_probs)), f'Chosen: {choice_per_prompt[0]}')

    return choice_per_prompt


def run_inference_rank(query_l, choices_l, out_name, args):
    batch = []
    choices_batch = []
    num_choices_q = []

    with out_name.open('w') as f_out, tqdm(total=len(query_l)) as pbar:
        for i, (query, choices) in enumerate(zip(query_l, choices_l)):
            prompts_with_choices = [f'{query} Answer: {choice}' for choice in choices]

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
            pbar.update(len(num_choices_q))


if __name__ == "__main__":
    args = parser.parse_args()
    client = load_openai_client(args.api_key, args.org)

    args.out_path.mkdir(exist_ok=True, parents=True)

    territories, countries, queries = load_borderlines_hf(args.dataset_dir)

    # process English control
    query_l = territories['Query']
    choices_l = territories['Claimants']
    out_all = args.out_path / 'responses_mc_all.txt'

    run_inference_rank(query_l, choices_l, out_all, args)

    # process multilingual
    # smaller batch for non-Latin tokenization
    args.batch_size = min(20, args.batch_size)
    for lang, ds in queries.items():
        query_l = ds['Query_Native']
        choices_l = ds['Claimants_Native']
        out_path = args.out_path / f'responses_mc.{lang}'

        run_inference_rank(query_l, choices_l, out_path, args)

    print('done')
