import pandas as pd
import numpy as np
import argparse
from pathlib import Path
from spn.io.file import from_file

parser = argparse.ArgumentParser(description='MAP Benchmark Results Plotter')
parser.add_argument('-id', '--experiment-ids', nargs='+',
    help='IDs of experiment results that you want to be included in the table')

# Parse input arguments and load results
args = parser.parse_args()
all_results = pd.read_csv('benchmark_results.csv')
assert args.experiment_ids, (
    "You need to provide a list of experiment ids"
)
experiment_ids = args.experiment_id
all_results = all_results[all_results["Experiment ID"] == exp_id]

results = all_results.groupby(
    ['Dataset', 'Method'])['MAP Probability'].agg(
    Mean_MAP_Probability='mean',
    Std_MAP_Probability='std'
).reset_index()
# results.to_csv('benchmark_results_aggregated.csv', index=False)

# Convert prob results to wide format
results_wide = results.pivot(
    index='Dataset', 
    columns='Method', 
    values=['Mean_MAP_Probability', 'Std_MAP_Probability']
)

# Find dimensionality of each dataset
script_dir = Path(__file__).parent
project_root = script_dir.parent
dataset_path = project_root / Path(dataset_name)
dataset_info = []
for spn_file in dataset_path.glob('*/*0.4q_0.6e.spn'):
    spn = from_file(spn_file)
    dataset_info.append({
        'Dataset': spn_file.parent.name,
        'Dimension': spn.vars(),
        'SPN Arcs': spn.arcs(),
        'SPN Nodes': spn.nodes()
    })
dim_df = pd.DataFrame(dataset_info)

# Produce summary of results based on ranks of each method's prob
def find_all_ranking(row):
    """
    Find method's rankings based on their Mean_MAP_Probability for each
    dataset.
    """
    mean_probs = row['Mean_MAP_Probability']
    
    # Use log probabilities for better numerical stability with small values
    # Add small epsilon to avoid log(0)
    log_probs = np.log(mean_probs + 1e-300)
    
    # Round in log space to 10 decimal places
    log_probs_rounded = log_probs.round(10)
    
    # Rank with method='min' gives same rank to ties
    # Higher log prob = higher rank (ascending=False)
    ranks = log_probs_rounded.rank(method='min', ascending=False)
    return ranks.values.tolist()

results_wide['All_Ranks'] = results_wide.apply(find_all_ranking, axis=1)
methods = results['Method'].unique()
rank_matrix = pd.DataFrame(
    results_wide['All_Ranks'].tolist(),
    columns=methods,
    index=results_wide.index
)
results_summary = pd.DataFrame({
    'Method': methods,
    'Average_Rank': rank_matrix.mean().round(4),
    'Median_Rank': rank_matrix.median().round(4),
    'Std_Rank': rank_matrix.std().round(4),
    'Times_Ranked_1st': (rank_matrix == 1).sum(),
    'Times_Ranked_Last': (rank_matrix == len(methods)).sum()
}).sort_values('Average_Rank')

# Combine mean and std into single strings
for method in results['Method'].unique():
    results_wide[method] = (
        results_wide['Mean_MAP_Probability'][method].map('{:.5g}'.format) + 
        ' +/- ' + 
        results_wide['Std_MAP_Probability'][method].map('{:.5g}'.format)
    )

# Keep only the combined columns, add columns on dataset dimension etc.
results_wide = results_wide[[method for method in results['Method'].unique()]]
results_wide = results_wide.reset_index()
results_wide = results_wide.merge(dim_df, on='Dataset', how='left')
results_wide.drop(columns='Dataset', inplace=True)
# results_wide['Times Sample_Cap Reached'] = [7, 0, 10, 10, 10, 0, 10, 10, 0, 10, 0, 10, 0, 10, 0, 10, 0, 1, 10, 0]

# Save the results to csvs
print(results_wide)
if args.date:
    datetime_folder = Path('results') / datetime_str
    datetime_folder.mkdir(parents=True, exist_ok=True)
    results_wide.to_csv(
        datetime_folder / f'{dataset_name}_results_{q_percent}q{e_percent}e_{datetime_str}.csv', 
        index=False
    )
    runtime_results_wide.to_csv(
        datetime_folder / f'{dataset_name}_runtimes_{q_percent}q{e_percent}e_{datetime_str}.csv', 
        index=False
    )
    summary.to_csv(
        datetime_folder / f'{dataset_name}_summary_{q_percent}q{e_percent}e_{datetime_str}.csv',
        index=False
    )
else:
    datetime_folder = Path('results') / exp_id
    datetime_folder.mkdir(parents=True, exist_ok=True)
    results_wide.to_csv(
        datetime_folder / f'{dataset_name}_results_{q_percent}q{e_percent}e_{exp_id}.csv', 
        index=False
    )
    runtime_results_wide.to_csv(
        datetime_folder / f'{dataset_name}_runtimes_{q_percent}q{e_percent}e_{exp_id}.csv', 
        index=False
    )
    summary.to_csv(
        datetime_folder / f'{dataset_name}_summary_{q_percent}q{e_percent}e_{exp_id}.csv',
        index=False
    )