import argparse
import ast
import json
from collections import Counter
from copy import copy
from itertools import combinations
from pathlib import Path
from statistics import mean

import datasets
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument(
    "input_paths", type=Path, nargs="+", help="Path to the CSV files with responses"
)
parser.add_argument("--output_path", "-o", type=Path)
parser.add_argument("--extended", "-v", action="store_true")
parser.add_argument("--info_path", type=Path)
parser.add_argument("--quiet", "-q", action="store_true")


def load_response_table(path):
    df = pd.read_csv(path)
    df["Claimants"] = df["Claimants"].str.split(";")
    df["Claimant_Codes"] = df["Claimant_Codes"].str.split(";")
    df["Responses_d"] = df["Responses_d"].apply(ast.literal_eval)
    return df


def fix_responses_d(df):
    res_ds_new = []
    for i, row in df.iterrows():
        res_d = copy(row["Responses_d"])
        if "en" not in row["Claimant_Codes"]:
            res_d.pop("en", None)
        res_ds_new.append(res_d)

    return pd.Series(res_ds_new)


def get_responses_d_en(df):
    res_ds_new = []
    for i, row in df.iterrows():
        res_d = copy(row["Responses_d"])
        res_d["en"] = row["Response_en"]
        res_ds_new.append(res_d)

    return pd.Series(res_ds_new)


def calculate_CS(df: pd.DataFrame, fix_responses_d, get_responses_d_en) -> dict:
    df["Responses_d"] = fix_responses_d(df)
    df["Unique_Claimants"] = df["Responses_d"].apply(lambda x: set(x.values()))

    df["Responses_d_en"] = get_responses_d_en(df)
    df["All_Claimants"] = df["Responses_d"].apply(lambda x: x.values())

    control_cs_l, non_cs_l, kb_cs_l, cons_cs_l = [], [], [], []

    for i, row in df.iterrows():
        controller = row["Controller"]
        controller_code = row["Controller_Code"]

        kb_cs = controller == row["Response_en"]

        non_cs = []
        res_d = row["Responses_d"]
        control_cs = None
        for l_code, claim in res_d.items():
            if controller != "Unknown":
                if l_code == controller_code:
                    control_cs = controller == claim
                else:
                    non_cs.append(controller == claim)

        pairs = list(combinations(row["All_Claimants"], 2))
        cons_cs = [x[0] == x[1] for x in pairs]

        if controller != "Unknown":
            control_cs_ref = (
                (row["Response_Controller"] == controller)
                if pd.notna(row["Response_Controller"])
                else None
            )
            assert control_cs == control_cs_ref
        else:
            kb_cs = None

        control_cs_l.append(control_cs)
        non_cs_l.append(non_cs)
        kb_cs_l.append(kb_cs)
        cons_cs_l.append(cons_cs)

    df["Control_CS"] = control_cs_l
    df["Non_CS"] = non_cs_l
    df["KB_CS"] = kb_cs_l
    df["Cons_CS"] = cons_cs_l

    n_rows = len(df)

    metrics_d = {}
    metrics_d["num_queries"] = int(df["Responses_d_en"].transform(len).sum())
    metrics_d["num_territories"] = n_rows

    # kb CS
    kb_cs_l = [x for x in kb_cs_l if x is not None]
    mean_kb_cs = mean(kb_cs_l)
    kb_frac = f"{kb_cs_l.count(True)}/{len(kb_cs_l)}"
    metrics_d["KB CS"] = mean_kb_cs
    metrics_d["KB CS Frac"] = kb_frac

    # control CS
    control_cs_l = [x for x in control_cs_l if x is not None]
    mean_control_cs = mean(control_cs_l)
    cs_frac = f"{control_cs_l.count(True)}/{len(control_cs_l)}"
    metrics_d["Control CS"] = mean_control_cs
    metrics_d["Control CS Frac"] = cs_frac

    # non-control CS
    non_cs_l = [x for x in non_cs_l if x != []]
    non_cs_means = [mean(subl) for subl in non_cs_l]
    mean_non_cs = mean(non_cs_means)
    metrics_d["Non-control CS"] = mean_non_cs

    delta = mean_control_cs - mean_non_cs
    delta_cs = delta / mean_non_cs
    metrics_d["Delta CS"] = delta_cs
    metrics_d["Delta CS unnormalized"] = delta

    # consistency CS for unknowns
    unk_inds = df[df["Controller"] == "Unknown"].index.tolist()
    cons_unk_cs_l = [x for i, x in enumerate(cons_cs_l) if i in unk_inds]
    cons_unk_cs_l = [x for x in cons_unk_cs_l if x != []]
    cons_unk_cs_means = [mean(subl) for subl in cons_unk_cs_l]
    mean_unk_cons_cs = mean(cons_unk_cs_means)
    metrics_d["Consistency CS unk"] = mean_unk_cons_cs
    metrics_d["Consistency CS unk rows"] = len(cons_unk_cs_l)

    # consistency CS
    cons_cs_l = [x for x in cons_cs_l if x != []]
    cons_cs_means = [mean(subl) for subl in cons_cs_l]
    mean_cons_cs = mean(cons_cs_means)
    metrics_d["Consistency CS all"] = mean_cons_cs
    metrics_d["Consistency CS all rows"] = len(cons_cs_l)

    # mean # for responses
    response_countries = df["Responses_d"].apply(lambda x: set(x.values())).transform(len)
    rcm, rcs = response_countries.mean(), response_countries.std()
    metrics_d["Response Countries mean"] = rcm
    metrics_d["Response Countries σ"] = rcs

    response_en_countries = df["Responses_d_en"].apply(lambda x: set(x.values())).transform(len)
    recm, recs = response_en_countries.mean(), response_en_countries.std()
    metrics_d["Response Countries + en mean"] = recm
    metrics_d["Response Countries + en σ"] = recs

    return metrics_d


def print_summary(metrics_d, df):
    mean_kb_cs = metrics_d.get("KB CS")
    kb_frac = metrics_d.get("KB CS Frac")
    mean_control_cs = metrics_d.get("Control CS")
    cs_frac = metrics_d.get("Control CS Frac")
    mean_non_cs = metrics_d.get("Non-control CS")
    delta_cs = metrics_d.get("Delta CS")
    delta = metrics_d.get("Delta CS unnormalized")
    mean_unk_cons_cs = metrics_d.get("Consistency CS unk")
    cons_unk_cs_l = metrics_d.get("Consistency CS unk rows")
    mean_cons_cs = metrics_d.get("Consistency CS all")
    cons_cs_l = metrics_d.get("Consistency CS all rows")
    rcm = metrics_d.get("Response Countries mean")
    rcs = metrics_d.get("Response Countries σ")
    recm = metrics_d.get("Response Countries + en mean")
    recs = metrics_d.get("Response Countries + en σ")
    n_rows = metrics_d.get("num_territories")
    n_queries_list = metrics_d.get("num_queries list")

    print("--- Means ---")

    if n_queries_list:
        # if not all queries have the same number of responses, print a warning
        if not all(x == n_queries_list[0] for x in n_queries_list):
            print("WARNING: Not all queries have the same number of responses. ")
            print(f"# queries per run: {n_queries_list}")
        else:
            print(f"All runs have {n_queries_list[0]} queries")
        kb_cs_stdev = metrics_d.get("KB CS σ")
        control_cs_stdev = metrics_d.get("Control CS σ")
        non_cs_stdev = metrics_d.get("Non-control CS σ")
        unk_cons_cs_stdev = metrics_d.get("Consistency CS unk σ")
        cons_cs_stdev = metrics_d.get("Consistency CS all σ")
        delta_cs_stdev = metrics_d.get("Delta CS σ")
        delta_stdev = metrics_d.get("Delta CS unnormalized σ")

        print(f"KB CS:\t\t\t{mean_kb_cs:.1%} (σ {kb_cs_stdev:.1%})")
        print(f"Control CS:\t\t{mean_control_cs:.1%} (σ {control_cs_stdev:.1%})")
        print(f"Non-control CS:\t\t{mean_non_cs:.1%} (σ {non_cs_stdev:.1%})")
        print(f"Delta CS:\t\t{delta_cs:.1%} (σ {delta_cs_stdev:.1%})")
        print(f"Delta CS abs:\t\t{delta:.1%} (σ {delta_stdev:.1%})")
        print(
            f"Consistency CS (unk):\t{mean_unk_cons_cs:.1%} (σ {unk_cons_cs_stdev:.1%}, over {cons_unk_cs_l} rows)"
        )
        print(
            f"Consistency CS (all):\t{mean_cons_cs:.1%} (σ {cons_cs_stdev:.1%}, over {cons_cs_l} rows)"
        )
        print(f"Mean # Response Countries: {rcm:.2f} (σ {rcs:.2f})")
        print(f"Mean # Response Countries + en: {recm:.2f} (σ {recs:.2f})")

        if args.extended:
            print("--- Full lists ---")
            longest_field = "Consistency CS (all) list"

            # print the lists for each metric
            def print_percentage_list(name, lst):
                print(f"{name:<{len(longest_field)}}: ", end="")
                for x in lst:
                    print(f"{x:.1%}, ", end="")
                print()

            print_percentage_list("KB CS list", metrics_d["KB CS list"])
            print_percentage_list("Control CS list", metrics_d["Control CS list"])
            print_percentage_list("Non-control CS list", metrics_d["Non-control CS list"])
            print_percentage_list("Delta CS list", metrics_d["Delta CS list"])
            print_percentage_list("Delta CS abs list", metrics_d["Delta CS unnormalized list"])
            print_percentage_list("Consistency CS (unk) list", metrics_d["Consistency CS unk list"])
            print_percentage_list(longest_field, metrics_d["Consistency CS all list"])
    else:
        print(
            f"KB CS:\t\t\t{mean_kb_cs:.1%} ({kb_frac})",
            f" -- filtered {n_rows - int(kb_frac.split('/')[1])} empty" if args.extended else "",
        )
        print(
            f"Control CS:\t\t{mean_control_cs:.1%} ({cs_frac})",
            f" -- filtered {n_rows - int(cs_frac.split('/')[1])} empty" if args.extended else "",
        )
        print(
            f"Non-control CS:\t\t{mean_non_cs:.1%}",
            f" -- filtered {n_rows - len([x for x in df['Non_CS'] if x != []])} empty"
            if args.extended
            else "",
        )
        print(f"Delta CS:\t\t{delta_cs:.1%}")
        print(f"Delta CS abs:\t\t{delta:.1%}")
        print(f"Consistency CS (unk):\t{mean_unk_cons_cs:.1%} (over {cons_unk_cs_l} rows)")
        print(f"Consistency CS (all):\t{mean_cons_cs:.1%} (over {cons_cs_l} rows)")
        print(f"Mean # Response Countries: {rcm:.2f} (σ {rcs:.2f})")
        print(f"Mean # Response Countries + en: {recm:.2f} (σ {recs:.2f})")

        if args.extended:
            # query-level stats
            claimants_list = [x for subl in df["Claimants"] for x in subl]
            counter = Counter(claimants_list)
            print(counter.most_common(2))
            counter_values = list(counter.values())
            print(
                f"Mean # territories per lang: {np.mean(counter_values):.2f} ({np.std(counter_values):.2f})"
            )

            claim_langs = df["Claimant_Codes"].apply(set).transform(len)
            print(f"Mean # Claimant Languages: {claim_langs.mean():.2f} ({claim_langs.std():.2f})")
            claims = df["Claimants"].apply(set).transform(len)
            print(f"Mean # Claimants: {claims.mean():.2f} ({claims.std():.2f})")

            print(
                f"Total # prompts: {df['Responses_d'].transform(len).sum()}",
                f" | Total # territories: {df.shape[0]}",
            )


def is_numeric(elem):
    return isinstance(elem, (int, float))


if __name__ == "__main__":
    args = parser.parse_args()
    if len(args.input_paths) == 1:
        df = load_response_table(args.input_paths[0])
        metrics_d = calculate_CS(df, fix_responses_d, get_responses_d_en)
        if not args.quiet:
            print_summary(metrics_d, df)
    else:
        metrics_list_d = {}
        for input_path in args.input_paths:
            df = load_response_table(input_path)
            metrics_d = calculate_CS(df, fix_responses_d, get_responses_d_en)

            for k, v in metrics_d.items():
                if k not in metrics_list_d:
                    metrics_list_d[k] = []
                metrics_list_d[k].append(v)

        metrics_d = {}
        for k, v_list in metrics_list_d.items():
            if is_numeric(v_list[0]):
                metrics_d[f"{k} list"] = v_list
                metrics_d[f"{k}"] = np.mean(v_list)
                metrics_d[f"{k} σ"] = np.std(v_list)
            else:
                metrics_d[k] = v_list

        if not args.quiet:
            print_summary(metrics_d, df)

    if args.output_path:
        with args.output_path.open("w") as f:
            json.dump(metrics_d, f, indent=2)
