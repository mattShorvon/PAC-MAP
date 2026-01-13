import pandas as pd
import numpy as np
import argparse
from pathlib import Path
from spn.io.file import from_file
from datetime import datetime

def get_method_colors():
    """Define colors for each method using ggsci NPG palette."""
    # NPG (Nature Publishing Group) palette from ggsci
    return {
        'ArgMax Product': 'npgRed',      # #E64B35
        'Independent': 'npgPurple',      # #8491B4
        'Max Product': 'npgBlue',        # #4DBBD5
        'PAC_MAP': 'npgGreen',           # #00A087
        'PAC_MAP_Hamming': 'npgNavy'     # #3C5488
    }


def format_rank_list_with_colors(rank_list, method_order=None):
    """
    Format list of (rank, method) tuples with LaTeX colors.
    
    Args:
        rank_list: List of (rank, method) tuples
        method_order: Not used (kept for compatibility)
    """
    if rank_list is None or (isinstance(rank_list, float) and np.isnan(rank_list)):
        return '-'
    
    method_colors = get_method_colors()
    formatted = []
    
    # Handle list of ranks
    for i, rank in enumerate(rank_list):
        if rank is None:
            formatted.append('-')
        else:
            method = method_order[i] if i < len(method_order) else None
            color = method_colors.get(method, 'black') if method else 'black'
            # Format: \textcolor{blue}{1.0}
            formatted.append(f"\\textcolor{{{color}}}{{{rank:.1f}}}")
    
    return ', '.join(formatted)

def create_colored_latex_table(rankings_df, method_order, caption, label):
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
    latex_str = r"\begin{table*}[ht]" + "\n"
    latex_str += r"\centering" + "\n"
    latex_str += r"\footnotesize" + "\n"
    latex_str += f"\\caption{{{caption}}}\n"
    latex_str += f"\\label{{{label}}}\n"
    
    # Define ggsci NPG colors
    latex_str += r"% Define ggsci NPG palette colors" + "\n"
    latex_str += r"\definecolor{npgRed}{HTML}{E64B35}" + "\n"
    latex_str += r"\definecolor{npgBlue}{HTML}{4DBBD5}" + "\n"
    latex_str += r"\definecolor{npgGreen}{HTML}{00A087}" + "\n"
    latex_str += r"\definecolor{npgNavy}{HTML}{3C5488}" + "\n"
    latex_str += r"\definecolor{npgPurple}{HTML}{8491B4}" + "\n"
    latex_str += r"\definecolor{npgOrange}{HTML}{F39B7F}" + "\n"
    latex_str += "\n"
    
    latex_str += r"\begin{tabularx}{\textwidth}{l|XXXX}" + "\n"
    latex_str += r"\toprule" + "\n"
    
    # Header
    latex_str += "Dataset & 10\\% Query & 25\\% Query & 50\\% Query & Dimension \\\\\n"
    latex_str += r"\midrule" + "\n"
    
    # Data rows
    for _, row in rankings_df_formatted.iterrows():
        dataset = row['Dataset']
        vals = [str(row[col]) for col in val_columns]
        if dataset == "pumsb_star":
            latex_str += r"\texttt{pumsb\_star} & " + ' & '.join(vals) + r" \\" + "\n"
        elif dataset == "ocr_letters":
            latex_str += r"\texttt{ocr\_letters} & " + ' & '.join(vals) + r" \\" + "\n"
        else:
            latex_str += r"\texttt{" + f"{dataset}" + r"} & " + ' & '.join(vals) + r" \\" + "\n"

    # Summary row
    latex_str += r"\midrule" + "\n"
    latex_str += r"No. Times Ranked Highest: & \textcolor{npgRed}{10}, \textcolor{npgPurple}{7}, \textcolor{npgBlue}{5}, \textcolor{npgGreen}{15}, \textcolor{npgNavy}{17} & \textcolor{npgRed}{11}, \textcolor{npgPurple}{1}, \textcolor{npgBlue}{4}, \textcolor{npgGreen}{12}, \textcolor{npgNavy}{14} & \textcolor{npgRed}{14}, \textcolor{npgPurple}{0}, \textcolor{npgBlue}{4}, \textcolor{npgGreen}{8}, \textcolor{npgNavy}{10} & \\" + "\n"

    latex_str += r"\bottomrule" + "\n"
    latex_str += r"\end{tabularx}" + "\n"

    # Legend with ggsci colors
    latex_str += r"\begin{center}" + "\n" 
    latex_str += r"\footnotesize" + "\n" 
    latex_str += r"\textcolor{npgRed}{ArgMax Product} \quad" + "\n"
    latex_str += r"\textcolor{npgPurple}{Independent} \quad" + "\n"
    latex_str += r"\textcolor{npgBlue}{Max Product} \quad" + "\n"
    latex_str += r"\textcolor{npgGreen}{PAC-MAP} \quad" + "\n"
    latex_str += r"\textcolor{npgNavy}{PAC-MAP-Smooth}" + "\n"
    latex_str += r"\end{center}" + "\n"
    latex_str += r"\end{table*}" + "\n"
    
    # Add package requirements at the top
    latex_note = r"% Requires \usepackage{xcolor} and \usepackage{tabularx} in your LaTeX preamble" + "\n"
    
    # Add legend
    method_colors = get_method_colors()
    legend = "\n% Legend (methods in alphabetical order, using ggsci NPG palette):\n"
    legend += "% NPG Colors: Red=#E64B35, Blue=#4DBBD5, Green=#00A087, Navy=#3C5488, Purple=#8491B4\n"
    for method, color in method_colors.items():
        legend += f"% {method}: \\textcolor{{{color}}}{{colored}}\n"
    
    return latex_note + legend + "\n" + latex_str

parser = argparse.ArgumentParser(description='MAP Benchmark Results Plotter')
parser.add_argument('-ids', '--experiment-ids', nargs='+',
    help='IDs of experiment results that you want to be included in the table')

# Parse input arguments and load results
args = parser.parse_args()
all_results = pd.read_csv('benchmark_results.csv')
assert args.experiment_ids, (
    "You need to provide a list of experiment ids"
)
exp_ids = args.experiment_ids
all_results = all_results[all_results["Experiment ID"].isin(exp_ids)]
all_results = all_results.sort_values(
    ['Experiment ID', 'Dataset', 'Query']
).reset_index(drop=True)

# Get number of methods
num_methods = len(all_results['Method'].unique())

# Average ranking
def rank_by_map_est(df, n_methods=4):
    """
    Rank MAP Probability for every n_methods consecutive rows.
    Methods with the same MAP Estimate get the same rank.
    
    Args:
        df: DataFrame sorted by (Experiment ID, Dataset, Query)
        n_methods: Number of methods per query
    
    Returns:
        DataFrame with added 'Rank' column
    """
    df = df.copy()
    ranks = []
    
    # Process every n_methods rows
    for i in range(0, len(df), n_methods):
        chunk = df.iloc[i:i+n_methods].copy()
        
        # Convert MAP Estimate dict to string for comparison
        chunk['MAP_Estimate_Str'] = chunk['MAP Estimate'].astype(str)
        
        # Get unique estimates and their probabilities
        estimate_probs = chunk.groupby('MAP_Estimate_Str')['MAP Probability'].first()
        
        # Rank the unique estimates
        log_probs = np.log(estimate_probs + 1e-300)
        log_probs = log_probs.round(12)
        estimate_ranks = log_probs.rank(method='min', ascending=False)
        
        # Map ranks back to each row
        chunk['Rank'] = chunk['MAP_Estimate_Str'].map(estimate_ranks)
        
        ranks.extend(chunk['Rank'].tolist())
    
    df['Rank'] = ranks
    return df

ranks_by_map_est = rank_by_map_est(all_results, n_methods=num_methods)
ranks_for_david = ranks_by_map_est.drop(columns=['Date', 'Query', 'MAP Estimate', "MAP Probability", "Runtime", "Experiment ID", "Evid Proportion"])
ranks_for_david['Trial'] = ranks_for_david.groupby(
    ['Dataset', 'Query Proportion', 'Method']
).cumcount() + 1 
ranks_for_david.to_csv("results/ranks_for_david.csv")

# Aggregate before pivoting
ranks_aggregated = ranks_by_map_est.groupby(
    ["Experiment ID", "Dataset", "Method"]
).agg({"Rank": 'mean'}).reset_index()
rank1_counts = ranks_aggregated[ranks_aggregated['Rank'] == 1.0].groupby(
    ['Experiment ID', 'Method']
).size().reset_index(name='Rank_1_Count')

# Create rankings grid manually - alphabetical order
method_order = sorted(all_results['Method'].unique())
rankings_per_cell = []

for dataset in ranks_aggregated['Dataset'].unique():
    row_data = {'Dataset': dataset}
    
    for exp_id in exp_ids:
        # Get all methods for this dataset and experiment
        subset = ranks_aggregated[
            (ranks_aggregated['Dataset'] == dataset) & 
            (ranks_aggregated['Experiment ID'] == exp_id)
        ]
        
        if subset.empty:
            row_data[exp_id] = None
            continue
        
        # Create list of ranks in alphabetical method order
        subset_indexed = subset.set_index('Method')
        ranks_list = [
            subset_indexed.loc[method, 'Rank'] 
            if method in subset_indexed.index else None
            for method in method_order
        ]
        
        row_data[exp_id] = ranks_list
    
    rankings_per_cell.append(row_data)

rankings_df = pd.DataFrame(rankings_per_cell)
rankings_df = rankings_df.rename(columns={
    exp_ids[0]: '10% Query',
    exp_ids[1]: '25% Query',
    exp_ids[2]: '50% Query'
})
rankings_df['Dimension'] = [111, 123, 100, 100, 500, 126, 180, 100, 64, 190, 17, 294, 112, 500, 16, 128, 69, 163, 500, 135]
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
with open(Path('results') / f'benchmark_rankings_avg_{datetime_str}.tex', 'w') as f:
    f.write(latex_table)