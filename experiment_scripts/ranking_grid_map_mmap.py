import pandas as pd
import numpy as np
import argparse
from pathlib import Path
from datetime import datetime
import ast


def first_existing_path(*candidates):
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return path
    raise FileNotFoundError(
        f"None of these result files exist: {', '.join(str(candidate) for candidate in candidates)}"
    )


CONDITION_SPECS = [
    {
        "label": "20% Q 80% E",
        "file": "results/benchmark_results.csv",
        "fallback_file": "result_csvs/benchmark_results.csv",
        "query": 0.2,
        "evidence": 0.8,
    },
    {
        "label": "50% Q 50% E",
        "file": "results/benchmark_results.csv",
        "fallback_file": "result_csvs/benchmark_results.csv",
        "query": 0.50,
        "evidence": 0.50,
    },
    {
        "label": "20% Q 50% E 30% N",
        "file": "results/benchmark_results_mmap.csv",
        "fallback_file": "result_csvs/benchmark_results_mmap.csv",
        "query": 0.20,
        "evidence": 0.50,
    },
    {
        "label": "50% Q 30% E 20% N",
        "file": "results/benchmark_results_mmap.csv",
        "fallback_file": "result_csvs/benchmark_results_mmap.csv",
        "query": 0.50,
        "evidence": 0.30,
    },
]

DIMENSION_MAP = {
    'accidents': 111,
    'adult': 123,
    'baudio': 100,
    'bnetflix': 100,
    'book': 500,
    'connect4': 126,
    'dna': 180,
    'jester': 100,
    'kdd': 64,
    'kosarek': 190,
    'msnbc': 17,
    'msweb': 294,
    'mushrooms': 112,
    'nips': 500,
    'nltcs': 16,
    'ocr_letters': 128,
    'plants': 69,
    'pumsb_star': 163,
    'tmovie': 500,
    'tretail': 135,
}

def get_method_colors():
    """Define colors for each method using ggsci NPG palette."""
    # NPG (Nature Publishing Group) palette from ggsci
    return {
        'ArgMax Product': 'npgRed',      # #E64B35
        'Independent': 'npgPurple',      # #8491B4
        'Max Product': 'npgBlue',        # #4DBBD5
        'PAC_MAP': 'npgGreen',           # #00A087
        'PAC_MAP_Hamming': 'npgNavy',    # #3C5488
        'ITSELF': 'npgOrange',           # #F39B7F
        'Hybrid Belief-Propagation': 'npgBrown',      # #7E6148
    }


def format_rank_list_with_colors(rank_list, method_order=None):
    """
    Format list of ranks with LaTeX colors and \mnum/\msep for alignment.

    Each cell becomes:
      \mnum{$\textcolor{color}{<rank>}$}\msep\mnum{$\textcolor{color}{<rank>}$}...
    """
    if rank_list is None or (isinstance(rank_list, float) and np.isnan(rank_list)):
        # Whole cell empty
        return r"\mnum{--}"
    
    method_colors = get_method_colors()
    formatted = []
    
    # Find minimum rank (winner)
    valid_ranks = [r for r in rank_list if r is not None and not (isinstance(r, float) and np.isnan(r))]
    if not valid_ranks:
        return r"\mnum{--}"
    min_rank = min(valid_ranks)
    
    for i, rank in enumerate(rank_list):
        method = method_order[i] if method_order and i < len(method_order) else None
        color = method_colors.get(method, 'black') if method else 'black'

        # Missing/absent method for this dataset
        if rank is None or (isinstance(rank, float) and np.isnan(rank)):
            inner = rf"$\textcolor{{{color}}}{{--}}$"
            formatted.append(rf"\mnum{{{inner}}}")
            continue

        # Inner colored number in math mode
        if rank == min_rank:
            inner = rf"$\textcolor{{{color}}}{{\mathbf{{{rank:.1f}}}}}$"
        else:
            inner = rf"$\textcolor{{{color}}}{{{rank:.1f}}}$"

        # Fixed-width box for vertical alignment
        formatted.append(rf"\mnum{{{inner}}}")
    
    # Separate entries with \msep (no commas)
    return r"\msep".join(formatted)

def create_colored_latex_table(
    rankings_df,
    method_order,
    condition_method_order_map,
    rank1_counts,
    condition_order,
    caption,
    label,
):
    r"""
    Create LaTeX table with colored ranks.
    Requires \usepackage{xcolor} and \usepackage{tabularx} in your LaTeX document.
    """
    rankings_df_formatted = rankings_df.copy()
    
    # Format each experiment column with colors
    colour_columns = [col for col in rankings_df.columns if col != 'Dataset']
    val_columns = [col for col in rankings_df.columns if col != 'Dataset']
    for col in colour_columns:
        method_order_for_col = condition_method_order_map.get(col, method_order)
        rankings_df_formatted[col] = rankings_df_formatted[col].apply(
            lambda x, methods=method_order_for_col: format_rank_list_with_colors(x, methods)
        )
    
    # Build LaTeX manually with tabularx for wrapping
    latex_str = r"\begin{table*}" + "\n"
    latex_str += f"\caption{{{caption}}}\n"
    latex_str += r"\setlength{\linewidth}{\textwidth}" + "\n"
    latex_str += r"\setlength{\hsize}{\textwidth}" + "\n"
    latex_str += r"\centering" + "\n"
    latex_str += r"\scriptsize" + "\n"
    
    # Define ggsci NPG colors
    latex_str += r"% Define ggsci NPG palette colors" + "\n"
    latex_str += r"\definecolor{npgRed}{HTML}{E64B35}" + "\n"
    latex_str += r"\definecolor{npgBlue}{HTML}{4DBBD5}" + "\n"
    latex_str += r"\definecolor{npgGreen}{HTML}{00A087}" + "\n"
    latex_str += r"\definecolor{npgNavy}{HTML}{3C5488}" + "\n"
    latex_str += r"\definecolor{npgPurple}{HTML}{8491B4}" + "\n"
    latex_str += r"\definecolor{npgOrange}{HTML}{F39B7F}" + "\n"
    latex_str += r"\definecolor{npgBrown}{HTML}{7E6148}" + "\n"
    latex_str += r"\setlength{\tabcolsep}{2pt}" + "\n"
    latex_str += "\n"
    
    latex_str += r"\begin{tabular}{l| c @{\hspace{0.2em}}| c @{\hspace{0.2em}}| c @{\hspace{0.2em}}| c}" + "\n"
    latex_str += r"\toprule" + "\n"

    # Header
    latex_str += "Dataset - Dimension & 20\\% Q 80\\% E & 50\\% Q 50\\% E & 20\\% Q 50\\% E 30\\% N & 50\\% Q 30\\% E 20\\% N \\\\\n"
    latex_str += r"\midrule" + "\n"
    
    # Data rows
    for _, row in rankings_df_formatted.iterrows():
        dataset = row['Dataset']
        vals = [str(row[col]) for col in val_columns]

        # Escape LaTeX-meaningful characters in dataset names
        dataset_tex = str(dataset).replace('_', r'\_')

        latex_str += r"\texttt{" + dataset_tex + r"} & " + ' & '.join(vals) + r" \\" + "\n"

    # --- Dynamic summary row: No. Times Ranked Highest ---
    latex_str += r"\midrule" + "\n"
    latex_str += r"\quad No. Times Ranked Highest: & "

    method_colors = get_method_colors()

    for k, condition in enumerate(condition_order):
        # Counts for this experiment
        counts_exp = rank1_counts[rank1_counts['Condition'] == condition]
        counts_map = {
            row['Method']: int(row['Best_Rank_Count'])
            for _, row in counts_exp.iterrows()
        }

        pieces = []
        for method in condition_method_order_map.get(condition, method_order):
            color = method_colors.get(method, 'black')
            count = counts_map.get(method, 0)
            inner = rf"$\textcolor{{{color}}}{{{count}}}$"
            pieces.append(rf"\mnum{{{inner}}}")

        latex_str += r"\msep".join(pieces)
        if k < len(condition_order) - 1:
            latex_str += " & "
        else:
            latex_str += r" \\" + "\n"

    latex_str += r"\bottomrule" + "\n"
    latex_str += r"\end{tabular}" + "\n"

    # --- Legend with current methods and colors ---
    latex_str += r"\begin{center}" + "\n" 
    latex_str += r"\scriptsize" + "\n" 
    latex_str += r"\textbf{Method:} \quad" + "\n"

    for method in method_order:
        color = method_colors[method]
        latex_str += rf"\textcolor{{{color}}}{{{method}}} \quad" + "\n"

    latex_str += r"\end{center}" + "\n"
    latex_str += f"\label{{{label}}}\n"
    latex_str += r"\end{table*}" + "\n"
    
    # Add package requirements at the top
    latex_note = r"% Requires \usepackage{xcolor} and \usepackage{tabularx} in your LaTeX preamble" + "\n"
    
    # Add legend comment (for source readability)
    legend = "\n% Legend (methods using ggsci NPG palette):\n"
    legend += (
        "% NPG Colors: Red=#E64B35, Blue=#4DBBD5, Green=#00A087, "
        "Navy=#3C5488, Purple=#8491B4, Orange=#F39B7F, Brown=#7E6148\n"
    )
    for method, color in method_colors.items():
        legend += f"% {method}: \\textcolor{{{color}}}{{colored}}\n"
    
    return latex_note + legend + "\n" + latex_str

parser = argparse.ArgumentParser(description='Combined MAP and MMAP ranking grid')
parser.add_argument('-ids', '--experiment-ids', nargs='+',
    help='Optional IDs of experiment results that you want to be included in the table')

# Parse input arguments and load results
args = parser.parse_args()

def load_and_rank_results(results_path, condition_specs):
    results_path = Path(results_path)
    all_results = pd.read_csv(results_path)

    if args.experiment_ids:
        all_results = all_results[all_results["Experiment ID"].astype(str).isin(args.experiment_ids)]

    all_results = all_results.sort_values(
        ['Query Proportion', 'Evid Proportion', 'Dataset', 'Query_ID', 'Method']
    ).reset_index(drop=True)

    ranked_frames = []
    for condition_spec in condition_specs:
        subset = all_results[
            (all_results['Query Proportion'] == condition_spec['query']) &
            (all_results['Evid Proportion'] == condition_spec['evidence'])
        ].copy()

        if subset.empty:
            continue

        subset['Condition'] = condition_spec['label']

        def _rank_group(group: pd.DataFrame) -> pd.DataFrame:
            group = group.copy()
            log_probs = np.log(group['MAP Probability'] + 1e-50).round(5)
            group['Rank'] = log_probs.rank(method='min', ascending=False)
            return group

        subset = subset.groupby(
            ['Condition', 'Dataset', 'Query_ID'], group_keys=False
        ).apply(_rank_group)

        ranked_frames.append(subset)

    if not ranked_frames:
        raise ValueError(f"No rows matched the requested conditions in {results_path}")

    return pd.concat(ranked_frames, ignore_index=True)

def _safe_ast(s):
    if pd.isna(s):
        return None
    try:
        return ast.literal_eval(str(s))
    except Exception:
        return s  # fallback: leave as-is

map_results = load_and_rank_results(
    first_existing_path('results/benchmark_results.csv', 'result_csvs/benchmark_results.csv'),
    CONDITION_SPECS[:2],
)
mmap_results = load_and_rank_results(
    first_existing_path('results/benchmark_results_mmap.csv', 'result_csvs/benchmark_results_mmap.csv'),
    CONDITION_SPECS[2:],
)

all_ranked_results = pd.concat([map_results, mmap_results], ignore_index=True)

# Aggregate before pivoting
ranks_aggregated = all_ranked_results.groupby(
    ["Condition", "Dataset", "Method"]
).agg({"Rank": 'mean'}).reset_index()
best_rank_mask = ranks_aggregated["Rank"] == ranks_aggregated.groupby(
    ["Condition", "Dataset"]
)["Rank"].transform("min")
rank1_counts = (
    ranks_aggregated.loc[best_rank_mask]
    .groupby(["Condition", "Method"])
    .size()
    .reset_index(name="Best_Rank_Count")
)

# Create rankings grid manually
method_order = list(get_method_colors().keys())
condition_order = [spec['label'] for spec in CONDITION_SPECS]
mmap_condition_labels = {spec['label'] for spec in CONDITION_SPECS[2:]}

condition_method_order_map = {}
for condition in condition_order:
    if condition in mmap_condition_labels:
        condition_method_order_map[condition] = [
            method for method in method_order if method != 'ITSELF'
        ]
    else:
        condition_method_order_map[condition] = list(method_order)

rankings_per_cell = []
dataset_order = [
    'accidents',
    'adult',
    'baudio',
    'bnetflix',
    'book',
    'connect4',
    'dna',
    'jester',
    'kdd',
    'kosarek',
    'msnbc',
    'msweb',
    'mushrooms',
    'nips',
    'nltcs',
    'ocr_letters',
    'plants',
    'pumsb_star',
    'tmovie',
    'tretail'
]
for dataset in dataset_order:
    if dataset not in ranks_aggregated['Dataset'].unique():
        print(f"Dataset {dataset} is not in this results set, moving to next")
        continue
    dimension = DIMENSION_MAP.get(dataset)
    dataset_label = f"{dataset} - {dimension}" if dimension is not None else dataset
    row_data = {'Dataset': dataset_label}

    for condition in condition_order:
        subset = ranks_aggregated[
            (ranks_aggregated['Dataset'] == dataset) &
            (ranks_aggregated['Condition'] == condition)
        ]

        if subset.empty:
            row_data[condition] = None
            continue

        method_order_for_condition = condition_method_order_map.get(condition, method_order)
        subset_indexed = subset.set_index('Method')
        ranks_list = [
            subset_indexed.loc[method, 'Rank']
            if method in subset_indexed.index else None
            for method in method_order_for_condition
        ]

        row_data[condition] = ranks_list

    rankings_per_cell.append(row_data)

rankings_df = pd.DataFrame(rankings_per_cell)

rankings_df = rankings_df[['Dataset'] + condition_order]
print(rankings_df)
print(f"\nMethod order: {method_order}")
print(f"Method order by condition: {condition_method_order_map}")

datetime_str = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
rankings_df.to_csv(
    Path('results') / f'benchmark_rankings_avg_{datetime_str}.csv',
    index=False
)

# Convert to LaTeX
# Create colored LaTeX table
latex_table = create_colored_latex_table(
    rankings_df,
    method_order,
    condition_method_order_map,
    rank1_counts,
    condition_order,
    caption='Average rankings of methods by MAP and MMAP estimates across ten trials. Winning entries in bold. Asterisks denote at least one timeout (see text).',
    label='tab:rankings_mapmmap'
)

print("\n" + "="*80)
print("LATEX TABLE (Average Rank - MAP/MMAP) - WITH COLORS:")
print("="*80)
print(latex_table)
print("="*80 + "\n")
print(f"Method order and colors:")
for method, color in get_method_colors().items():
    print(f"  {method}: {color}")

# Save to file
# with open(Path('results') / f'benchmark_pgms_rankings_avg_{datetime_str}.tex', 'w') as f:
#     f.write(latex_table)