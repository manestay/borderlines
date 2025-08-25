import argparse
from pathlib import Path

import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--input_paths", "-i", type=Path, nargs="+", required=True)
parser.add_argument("--output_path", "-o", type=Path, required=True)
parser.add_argument("--print_latex", action="store_true")


HEADER = [
    "Model",
    "Role",
    "KB CS",
    "Control CS",
    "Non-control CS",
    "Delta CS",
    "Delta CS unnormalized",
    "Consistency CS unk",
    "Consistency CS all",
    "Response Countries mean",
    "Response Countries \u03c3",
]
CS_COLS = [col for col in HEADER if "CS" in col]
IR_MODES = ["no_ir", "qlang", "qlang_en", "en", "rel_langs", "ablate_swap"]
IR_MODE_ORDER = {mode: i for i, mode in enumerate(IR_MODES)}


def get_config(stem):
    stem = stem.rsplit("_", 1)[0]  # remove _scores
    if stem.startswith("gpt4"):
        model, prompt_mode = stem.split("_", 1)
    else:
        model, prompt_mode = stem, "none"
    return {"Model": model, "Role": prompt_mode}


def gen_table(input_paths):
    import json

    rows = []
    for input_path in input_paths:
        stem = input_path.stem
        config = get_config(stem)

        with input_path.open() as f:
            data = json.load(f)
        data.update(config)
        rows.append(data)

    # add a random baseline row
    random_row = {"Model": "random", "Role": "none"}
    for col in CS_COLS:
        if col == "Delta CS":
            random_row[col] = 0.0
        else:
            random_row[col] = 0.435
    rows.append(random_row)

    df = pd.DataFrame(rows, columns=HEADER)
    for col in CS_COLS:
        df[col] = df[col] * 100
    return df


def main():
    args = parser.parse_args()
    df = gen_table(args.input_paths)
    sort_order = ["Model", "Role"]
    df = df.sort_values(by=sort_order, ignore_index=True)
    df.to_csv(args.output_path, index=False, float_format="%.3f")
    if args.print_latex:
        formatters = {col: (lambda x: f"{x:.1f}") for col in CS_COLS}
        formatters.update(
            {
                col: (lambda x: f"{x:.3f}")
                for col in ["Response Countries mean", "Response Countries \u03c3"]
            }
        )
        print(df.to_latex(index=False, formatters=formatters))


if __name__ == "__main__":
    main()
