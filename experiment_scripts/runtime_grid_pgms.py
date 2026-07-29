import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd


METHOD_SPECS = [
    ("DAOOPT-AOBB", "DAOOPT-AOBB", "npgRed"),
    ("DAOOPT-AOBF", "DAOOPT-AOBF", "npgBlue"),
    ("IJGP", "IJGP", "npgGreen"),
    ("WMB Elimination", "WMB", "npgBrown"),
    ("PAC_MAP", "PAC-MAP", "npgNavy"),
    ("PAC_MAP_Hamming", "smooth-PAC-MAP", "npgPurple"),
]


def first_existing_path(*candidates):
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return path
    return Path(candidates[-1])


def format_dataset_name(dataset):
    return r"\texttt{" + dataset.replace("_", r"\_") + "}"


def format_number(value):
    if value == 0:
        return "0"

    digits = int(math.floor(math.log10(abs(value))))
    decimal_places = max(0, 2 - digits)
    formatted = f"{round(value, decimal_places):.{decimal_places}f}"
    if "." in formatted:
        formatted = formatted.rstrip("0").rstrip(".")
    return formatted if formatted else "0"


def format_value(mean_value, std_value, color_name, is_best):
    mean_str = format_number(mean_value)
    std_str = format_number(std_value)
    if is_best:
        return rf"$\textcolor{{{color_name}}}{{\bm{{{mean_str}}}}} \pm {std_str}$"
    return rf"$\textcolor{{{color_name}}}{{{mean_str}}} \pm {std_str}$"


def resolve_trial_summary(df, q_percent, e_percent):
    mask = np.isclose(df["Query Proportion"], q_percent) & np.isclose(df["Evid Proportion"], e_percent)
    filtered = df.loc[mask].copy()
    if filtered.empty:
        available = (
            df[["Query Proportion", "Evid Proportion"]]
            .drop_duplicates()
            .sort_values(["Query Proportion", "Evid Proportion"])
        )
        raise ValueError(
            f"No benchmark rows found for q={q_percent} and e={e_percent}. "
            f"Available settings are: {available.to_dict('records')}"
        )

    summary = (
        filtered.groupby(["Dataset", "Method"], sort=False)["Runtime"]
        .agg(mean="mean", std="std")
        .reset_index()
    )
    return summary, filtered


def convert_runtimes_to_latex(csv_path, output_path=None, q_percent=0.1, e_percent=0.9):
    """Convert PGM benchmark runtime results into a compact LaTeX table."""
    df = pd.read_csv(csv_path)
    summary, filtered = resolve_trial_summary(df, q_percent, e_percent)

    dataset_order = sorted(filtered["Dataset"].drop_duplicates().tolist())
    runtime_lookup = summary.set_index(["Dataset", "Method"])

    latex_lines = [
        r"% Requires \usepackage{xcolor}, \usepackage{booktabs}, \usepackage{bm}",
        r"\definecolor{npgRed}{HTML}{E64B35}",
        r"\definecolor{npgBlue}{HTML}{4DBBD5}",
        r"\definecolor{npgGreen}{HTML}{00A087}",
        r"\definecolor{npgNavy}{HTML}{3C5488}",
        r"\definecolor{npgPurple}{HTML}{8491B4}",
        r"\definecolor{npgOrange}{HTML}{F39B7F}",
        r"\definecolor{npgBrown}{HTML}{7E6148}",
        "",
        r"\begin{table*}[t]",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{2pt}",
        r"\renewcommand{\arraystretch}{0.96}",
        rf"\caption{{Runtime comparison (seconds) for {int(round(q_percent * 100))}\% query and {int(round(e_percent * 100))}\% evidence. We report the mean $\pm$ standard deviation over the available trials.}}",
        r"\label{tab:pgm_runtimes}",
        r"\begin{tabular}{@{}lcccccc@{}}",
        r"\toprule",
        "Dataset & " + " & ".join(short_name for _, short_name, _ in METHOD_SPECS) + r" \\",
        r"\midrule",
    ]

    for dataset in dataset_order:
        dataset_label = format_dataset_name(dataset)
        row_values = []
        row_means = []

        for method, _, color_name in METHOD_SPECS:
            if (dataset, method) not in runtime_lookup.index:
                row_values.append(r"--")
                row_means.append(np.nan)
                continue

            record = runtime_lookup.loc[(dataset, method)]
            mean_value = float(record["mean"])
            std_value = 0.0 if pd.isna(record["std"]) else float(record["std"])
            row_means.append(mean_value)
            row_values.append(format_value(mean_value, std_value, color_name, False))

        min_mean = np.nanmin(row_means)
        for idx, (method, _, color_name) in enumerate(METHOD_SPECS):
            if np.isnan(row_means[idx]):
                continue
            if np.isclose(row_means[idx], min_mean):
                record = runtime_lookup.loc[(dataset, method)]
                row_values[idx] = format_value(
                    float(record["mean"]),
                    0.0 if pd.isna(record["std"]) else float(record["std"]),
                    color_name,
                    True,
                )

        latex_lines.append(dataset_label + " & " + " & ".join(row_values) + r" \\")

    latex_lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
    ])

    latex_str = "\n".join(latex_lines) + "\n"
    print(latex_str)

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(latex_str)
        print(f"\nLaTeX table saved to {output_path}")

    return latex_str


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert PGM benchmark runtimes to a LaTeX table")
    parser.add_argument("-q", "--q-percent", type=float, required=True, help="Proportion of query variables")
    parser.add_argument("-e", "--e-percent", type=float, required=True, help="Proportion of evidence variables")
    parser.add_argument(
        "--csv-path",
        default=None,
        help="Benchmark CSV path. Defaults to result_csvs/benchmark_results_pgms.csv with a fallback to results/benchmark_results_pgms.csv.",
    )
    parser.add_argument(
        "--output-path",
        default=None,
        help="Optional output .tex path. Defaults to result_csvs/runtimes_table_pgms_<q>q<e>e.tex.",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv_path) if args.csv_path else first_existing_path(
        "result_csvs/benchmark_results_pgms.csv",
        "results/benchmark_results_pgms.csv",
        "benchmark_results_pgms.csv",
    )
    q_tag = int(round(args.q_percent * 100))
    e_tag = int(round(args.e_percent * 100))
    output_path = Path(args.output_path) if args.output_path else Path(
        f"result_csvs/runtimes_table_pgms_{q_tag}q{e_tag}e.tex"
    )

    convert_runtimes_to_latex(csv_path, output_path, args.q_percent, args.e_percent)