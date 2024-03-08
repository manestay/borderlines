# adapted from https://github.com/yuchenlin/rank_outputs/blob/main/rank_outputs/main.py

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer
from utils import DataCollatorForMultipleChoice, ModelBase, convert_features

from lib import load_borderlines_hf

LETTERS = 'ABCDEFG'

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_dir', '-dd', type=Path, help='path to dataset saved locally')
parser.add_argument('--out_dir', '-o', type=Path, required=True)
parser.add_argument('--model_name', '-m', default='bigscience/mt0-base')
parser.add_argument('--batch_size', '-b', type=int, default=10)
parser.add_argument('--load_in_4bit', '-fb', action='store_true')

def init_model(model_name_or_path, load_in_4bit):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    tokenizer.padding_side = 'right'
    config = AutoConfig.from_pretrained(model_name_or_path)

    kwargs = {"device_map": "auto"}
    if load_in_4bit:  # as model is loaded once, changing this arg later does nothing
        kwargs.update(
            {"load_in_4bit": True, 'bnb_4bit_compute_dtype': torch.float16})
    model = ModelBase.from_config(
            config=config,
            model_name_or_path=model_name_or_path,
            parallelize=True,
            **kwargs
        )
    return model, tokenizer

def format_data(prompts, choices_l, targets=None):
    # targets not used for ranking
    if not targets:
        targets = [''] * len(prompts)

    data = []

    max_num_choices = max(len(x) for x in choices_l)
    for prompt, choices, target in zip(prompts, choices_l, targets):
        # prompt = prompt + ' Answer :'
        diff = max_num_choices - len(choices)
        if diff != 0:
            choices.extend([f'<unused_{i}>' for i in range(diff)])
        item = {"input_texts": [prompt], "answer_choices_texts": [choices], "target_texts": [target]}
        data.append(item)
    return data

def run_query(prompts, choices_l, out_path, model, tokenizer, batch_size):
    max_num_choices = max(len(x) for x in choices_l)
    data = format_data(prompts, choices_l)
    features = [convert_features(tokenizer, item) for item in data]
    features = [convert_features(tokenizer, item) for item in data]
    data_collator = DataCollatorForMultipleChoice(
                tokenizer, pad_to_multiple_of=None, padding=True, max_length=64
                )
    eval_dataloader = DataLoader(features, collate_fn=data_collator, batch_size=batch_size)

    model.eval()
    with out_path.open('w') as f_out:
        for batch in tqdm(eval_dataloader):
            batch = {k: v.to('cuda') for k, v in batch.items()}
            last_dim = batch['labels'].shape[-1]
            labels_batch = batch['labels'].reshape(-1, max_num_choices, last_dim)
            choices_batch = [tokenizer.batch_decode(x, skip_special_tokens=True) \
                                for x in labels_batch]
            with torch.no_grad():
                predictions, seq_log_prob = model(batch)
                # print(predictions)
                # print(seq_log_prob)
            choice_per_line = [f'{LETTERS[pred_ind]}) {choices[pred_ind]}' \
                                for choices, pred_ind in zip(choices_batch, predictions)]
            f_out.writelines([x + '\n' for x in choice_per_line])
    print(f'wrote to {out_path}')


if __name__ == "__main__":
    args = parser.parse_args()

    if not args.out_dir.suffix == '.txt':
        args.out_dir.mkdir(exist_ok=True)
    else:
        args.out_dir.parent.mkdir(exist_ok=True)

    territories, countries, queries = load_borderlines_hf(args.dataset_dir)

    print(f'loading model {args.model_name}... ', end='')
    model, tokenizer = init_model(args.model_name, args.load_in_4bit)
    print(f'done')

    # process English setting
    print(f'processing English queries for all {len(territories)} territories')
    prompts = territories['Query']
    choices_l = territories['Claimants']
    out_path = args.out_dir / f'responses_mc_all.txt' if not args.out_dir.suffix else args.out_dir
    run_query(prompts, choices_l, out_path, model, tokenizer, args.batch_size)

    for lang, ds in queries.items():
        prompts = ds['Query_Native']
        choices_l = ds['Claimants_Native']
        out_path = args.out_dir / f'responses_mc.{lang}'
        run_query(prompts, choices_l, out_path, model, tokenizer, args.batch_size)
