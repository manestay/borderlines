import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoTokenizer
from utils import DataCollatorForMultipleChoice, ModelBase, convert_features

LETTERS = 'ABCDEFG'

parser = argparse.ArgumentParser()
parser.add_argument('prompts_dir', type=Path)
parser.add_argument('out_dir', type=Path)
parser.add_argument('--model_name', '-m', default='bigscience/mt0-base')
parser.add_argument('--batch_size', '-b', type=int, default=10)

def init_model(model_name_or_path):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    tokenizer.padding_side = 'right'
    config = AutoConfig.from_pretrained(model_name_or_path)
    model = ModelBase.from_config(
            config=config,
            model_name_or_path=model_name_or_path,
            parallelize=True
        )
    return model, tokenizer

def format_data(prompts, choices_l, targets=None):
    if not targets:
        targets = [''] * len(prompts)

    data = []

    max_num_choices = max(len(x) for x in choices_l)
    for prompt, choices, target in zip(prompts, choices_l, targets):
        # prompt = prompt + ' The answer is'
        diff = max_num_choices - len(choices)
        if diff != 0:
            choices.extend([f'<unused_{i}>' for i in range(diff)])
        item = {"input_texts": [prompt], "answer_choices_texts": [choices], "target_texts": [target]}
        data.append(item)
    return data



if __name__ == "__main__":

    args = parser.parse_args()

    if not args.out_dir.suffix == '.txt':
        args.out_dir.mkdir(exist_ok=True)
    else:
        args.out_dir.parent.mkdir(exist_ok=True)

    print(f'loading model {args.model_name}... ', end='')
    model, tokenizer = init_model(args.model_name)
    print(f'done')

    if args.prompts_dir.suffix == '.txt':
        prompt_paths = [args.prompts_dir]
    else:
        prompt_paths = args.prompts_dir.glob('prompts*')

    for prompt_path in prompt_paths:
        code = prompt_path.suffixes[0].lstrip('.')
        choices_path = prompt_path.parent / f'choices.{code}'
        if args.out_dir.suffix == '.txt':
            out_path = args.out_dir
        else:
            out_path = args.out_dir / f'responses_mc.{code}'

        print(f'processing {code}    ', end='\r')

        with prompt_path.open('r') as f:
            prompts = [x.strip() for x in f]

        with choices_path.open('r') as f:
            choices_l = [json.loads(x) for x in f]
        max_num_choices = max(len(x) for x in choices_l)

        data = format_data(prompts, choices_l)
        # for d in data:
        #     print(d)
        features = [convert_features(tokenizer, item) for item in data]
        data_collator = DataCollatorForMultipleChoice(
                    tokenizer, pad_to_multiple_of=None, padding=True, max_length=64
                    )
        eval_dataloader = DataLoader(features, collate_fn=data_collator, batch_size=args.batch_size)

        model.eval()
        with out_path.open('w') as f_out:
            for batch in eval_dataloader:
                batch = {k: v.to('cuda') for k, v in batch.items()}
                last_dim = batch['labels'].shape[-1]
                labels_batch = batch['labels'].reshape(-1, max_num_choices, last_dim)
                choices_batch = [tokenizer.batch_decode(x, skip_special_tokens=True) \
                                 for x in labels_batch]
                with torch.no_grad():
                    predictions, seq_log_prob = model(batch)
                    print(predictions)
                    # print(seq_log_prob)
                choice_per_line = [f'{LETTERS[pred_ind]}) {choices[pred_ind]}' \
                                   for choices, pred_ind in zip(choices_batch, predictions)]
                f_out.writelines([x + '\n' for x in choice_per_line])
