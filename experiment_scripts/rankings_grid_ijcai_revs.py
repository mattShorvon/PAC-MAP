import pandas as pd
import numpy as np
import argparse
from pathlib import Path
from spn.io.file import from_file
from datetime import datetime
import ast

def get_method_colors():
    """Define colors for each method using ggsci NPG palette."""
    # NPG (Nature Publishing Group) palette from ggsci
    return {
        'AAOBF': 'npgPurple',             # #8491B4
        'Hybrid Belief-Propagation': 'npgBrown',  # #7E6148
        'ITSELF': 'npgOrange',            # #F39B7F
        'PAC_MAP_Hamming': 'npgNavy',     # #3C5488
        'RBFAOO': 'npgGreen',             # #00A087
        'WMB Elimination': 'npgRed',      # #E64B35
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
    rank1_counts,
    exp_ids,
    caption,
    label,
):
    r"""
    Create LaTeX table with colored ranks.
    Requires \usepackage{xcolor} and \usepackage{tabularx} in your LaTeX document.
    """
    rankings_df_formatted = rankings_df.copy()
    
    # Format each experiment column with colors
    colour_columns = [col for col in rankings_df.columns if col != 'Dataset' and col != 'Dimension']
    val_columns = [col for col in rankings_df.columns if col != 'Dataset']
    for col in colour_columns:
        rankings_df_formatted[col] = rankings_df_formatted[col].apply(
            lambda x: format_rank_list_with_colors(x, method_order)
        )
    
    # Build LaTeX manually with tabularx for wrapping
    latex_str = r"\begin{table}[t!]" + "\n"
    latex_str += r"\setlength{\linewidth}{\textwidth}" + "\n"
    latex_str += r"\setlength{\hsize}{\textwidth}" + "\n"
    latex_str += r"\centering" + "\n"
    latex_str += r"\scriptsize" + "\n"
    latex_str += f"%\caption{{{caption}}}\n"
    latex_str += f"%\label{{{label}}}\n"
    
    # Define ggsci NPG colors
    latex_str += r"% Define ggsci NPG palette colors" + "\n"
    latex_str += r"\definecolor{npgRed}{HTML}{E64B35}" + "\n"
    latex_str += r"\definecolor{npgBlue}{HTML}{4DBBD5}" + "\n"
    latex_str += r"\definecolor{npgGreen}{HTML}{00A087}" + "\n"
    latex_str += r"\definecolor{npgNavy}{HTML}{3C5488}" + "\n"
    latex_str += r"\definecolor{npgPurple}{HTML}{8491B4}" + "\n"
    latex_str += r"\definecolor{npgOrange}{HTML}{F39B7F}" + "\n"
    latex_str += r"\definecolor{npgBrown}{HTML}{7E6148}" + "\n"
    latex_str += "\n"
    
    # latex_str += r"\begin{tabular}{l@{\hspace{0.1em}}l|lll}" + "\n"
    latex_str += r"\begin{tabular}{l @{\hspace{1em}} r|@{\hspace{1.2em}} c @{\hspace{2.0em}} c @{\hspace{2.0em}} c}" + "\n"
    latex_str += r"\toprule" + "\n"

    # Header
    latex_str += "Dataset & Dimension & 10\\% Query & 25\\% Query & 50\\% Query \\\\\n"
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
    latex_str += r"\quad No. Times Ranked Highest: & & "

    method_colors = get_method_colors()

    for k, exp_id in enumerate(exp_ids):
        # Counts for this experiment
        counts_exp = rank1_counts[rank1_counts['Query Proportion'] == exp_id]
        counts_map = {
            row['Method']: int(row['Rank_1_Count'])
            for _, row in counts_exp.iterrows()
        }

        pieces = []
        for method in method_order:
            color = method_colors.get(method, 'black')
            count = counts_map.get(method, 0)
            inner = rf"$\textcolor{{{color}}}{{{count}}}$"
            pieces.append(rf"\mnum{{{inner}}}")

        latex_str += r"\msep".join(pieces)
        if k < len(exp_ids) - 1:
            latex_str += " & "
        else:
            latex_str += r" \\" + "\n"

    latex_str += r"\bottomrule" + "\n"
    latex_str += r"\end{tabular}" + "\n"

    # --- Legend with current methods and colors ---
    latex_str += r"\begin{center}" + "\n" 
    latex_str += r"\scriptsize" + "\n" 
    latex_str += r"\textbf{Method:} \quad" + "\n"

    display_labels = {
        "WMB Elimination": "WMB",
        "Hybrid Belief-Propagation": "HBP",
        "PAC_MAP_Hamming": "smooth-PACMAP"
    }
    for method in method_order:
        color = method_colors[method]
        label = display_labels.get(method, method)
        latex_str += rf"\textcolor{{{color}}}{{{label}}} \quad" + "\n"

    latex_str += r"\end{center}" + "\n"
    latex_str += r"\end{table}" + "\n"
    
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

parser = argparse.ArgumentParser(description='MAP Benchmark Results Plotter')
parser.add_argument('-ids', '--experiment-ids', nargs='+',
    help='IDs of experiment results that you want to be included in the table')

# Parse input arguments and load results
args = parser.parse_args()
all_results = pd.read_csv('benchmark_results_ijcaireview.csv')
assert args.experiment_ids, (
    "You need to provide a list of experiment ids"
)
exp_ids = args.experiment_ids
all_results = all_results[all_results["Experiment ID"].isin(exp_ids)]

def _safe_ast(s):
    if pd.isna(s):
        return None
    try:
        return ast.literal_eval(str(s))
    except Exception:
        return s  # fallback: leave as-is

# Canonicalize Query to a sorted tuple representation
all_results['Query_norm'] = (
    all_results['Query']
    .apply(_safe_ast)
    .apply(lambda q: tuple(sorted(q)) if isinstance(q, (list, tuple)) else q)
)

# Now sort 
all_results = all_results.sort_values(
    ['Query Proportion', 'Dataset', 'Query_ID', 'Method']
).reset_index(drop=True)

# mask = (
#     (all_results["Dataset"] == "nltcs") &
#     (all_results["Query Proportion"] == 0.1) &
#     (all_results["Evid Proportion"] == 0.9)
# )
# good_cols = ['Method', 'MAP Estimate', 'MAP_Estimate_norm', 'MAP Probability', 'Query Proportion', 'Evid Proportion', 'Query_norm', 'Query', 'Rank']
# print(
#     all_results.loc[mask, good_cols]
#         .to_string()
# )

def rank_by_map_est(df):
    """
    Rank MAP Probability for each (Experiment ID, Dataset, Query_ID) group.
    Methods with the same MAP Estimate get the same rank.
    """
    df = df.copy()

    def _rank_group(group: pd.DataFrame) -> pd.DataFrame:
        group = group.copy()
        log_probs = np.log(group['MAP Probability'] + 1e-50).round(5)
        group['Rank'] = log_probs.rank(method='min', ascending=False)

        return group

    df = df.groupby(
        ['Query Proportion', 'Dataset', 'Query_ID'], group_keys=False
    ).apply(_rank_group)
    return df

ranks_by_map_est = rank_by_map_est(all_results)

# Aggregate before pivoting
ranks_aggregated = ranks_by_map_est.groupby(
    ["Query Proportion", "Dataset", "Method"]
).agg({"Rank": 'mean'}).reset_index()
rank1_counts = ranks_aggregated[ranks_aggregated['Rank'] == 1.0].groupby(
    ['Query Proportion', 'Method']
).size().reset_index(name='Rank_1_Count')

# Create rankings grid manually
# method_order = sorted(all_results['Method'].unique())
method_order = [
    'WMB Elimination',
    'AAOBF',
    'RBFAOO',
    'Hybrid Belief-Propagation',
    'ITSELF',
    'PAC_MAP_Hamming',  # smooth-PACMAP shown last
]
rankings_per_cell = []
q_proportions = ranks_aggregated['Query Proportion'].unique()
dataset_order = [
    'bnetflix',
    'connect4',
    'dna',
    'jester',
    'kosarek',
    'pumsb_star',
    'vote',
    'nltcs',
    'bridges_version1',
    'tae',
    'hayes-roth_train',
    'iris',
]
# for dataset in ranks_aggregated['Dataset'].unique():
for dataset in dataset_order:
    if dataset not in ranks_aggregated['Dataset'].unique():
        print(f"Dataset {dataset} is not in this results set, moving to next")
        continue 
    row_data = {'Dataset': dataset}
    
    for q in q_proportions:
        # Get all methods for this dataset and experiment
        subset = ranks_aggregated[
            (ranks_aggregated['Dataset'] == dataset) & 
            (ranks_aggregated['Query Proportion'] == q)
        ]
        
        if subset.empty:
            row_data[q] = None
            continue
        
        # Create list of ranks in alphabetical method order
        subset_indexed = subset.set_index('Method')
        ranks_list = [
            subset_indexed.loc[method, 'Rank'] 
            if method in subset_indexed.index else None
            for method in method_order
        ]
        
        row_data[q] = ranks_list
    
    rankings_per_cell.append(row_data)

rankings_df = pd.DataFrame(rankings_per_cell)
rankings_df = rankings_df.rename(columns={
    q_proportions[0]: '10% Query',
    q_proportions[1]: '25% Query',
    q_proportions[2]: '50% Query'
})
rankings_df['Dimension'] = [100, 126, 180, 100, 190, 163, 17, 16, 10,  6,  5, 5]
rankings_df = rankings_df[['Dataset', 'Dimension', '10% Query', '25% Query', '50% Query']]
print(rankings_df)
print(f"\nMethod order: {method_order}")

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
    rank1_counts,
    q_proportions,
    caption='Average ranking of methods by MAP estimate',
    label='tab:rankings_mapest'
)

print("\n" + "="*80)
print("LATEX TABLE (Average Rank - MAP Estimate) - WITH COLORS:")
print("="*80)
print(latex_table)
print("="*80 + "\n")
print(f"Method order and colors:")
for method, color in get_method_colors().items():
    print(f"  {method}: {color}")

# Save to file
with open(Path('ijcai_reviews') / f'benchmark_rankings_avg_{datetime_str}.tex', 'w') as f:
    f.write(latex_table)