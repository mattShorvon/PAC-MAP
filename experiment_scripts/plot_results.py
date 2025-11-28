import pandas as pd
import numpy as np
import argparse
from pathlib import Path

parser = argparse.ArgumentParser(description='MAP Benchmark Results Plotter')
parser.add_argument('-q', '--q-percent', type=float, default=0.1,
                    help="Proportion of query variables")
parser.add_argument('-e', '--e-percent', type=float, default=0.1,
                    help="Proportion of evidence variables")
parser.add_argument('-dn', '--dataset-name', default='small_datasets',
                    help='Name of the dataset group e.g. 20 datasets')
parser.add_argument('-dt', '--date', help='Date and time of experiment')

# Parse input arguments
args = parser.parse_args()
q_percent = args.q_percent
e_percent = args.e_percent
if args.date:
    datetime_str = args.date
else:
    datetime_str = "2025-11-05 20-02-33"
dataset_name = args.dataset_name

# Load results
all_results = pd.read_csv('benchmark_results.csv')
all_results = all_results[all_results["Date"] == datetime_str]
all_results = all_results[all_results["Query Proportion"] == q_percent]
all_results = all_results[all_results["Evid Proportion"] == e_percent]


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

# Produce summary of results based on ranks of each method's prob

def find_all_ranking(row):
    """
    Find method's rankings based on their Mean_MAP_Probability for each
    dataset.
    """
    # Round to avoid floating point comparison issues
    mean_probs = row['Mean_MAP_Probability'].round(10)

    # Rank with method='min' gives same rank to ties
    ranks = mean_probs.rank(method='min', ascending=False)
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

# Keep only the combined columns
results_wide = results_wide[[method for method in results['Method'].unique()]]
results_wide = results_wide.reset_index()

# Get the runtime results
runtime_results = all_results.groupby(['Dataset', 'Method'])['Runtime'].agg(
    Mean_Runtime= 'mean',
    Std_Runtime= 'std'
).reset_index()

runtime_results_wide = runtime_results.pivot(
    index='Dataset',
    columns='Method',
    values=['Mean_Runtime', 'Std_Runtime']
)

def find_runtime_ranking(row):
    """
    Find method's rankings based on their Mean_Runtime for each
    dataset.
    """
    # Round to avoid floating point comparison issues
    mean_probs = row['Mean_Runtime'].round(10)

    # Rank with method='min' gives same rank to ties
    ranks = mean_probs.rank(method='min', ascending=True)
    return ranks.values.tolist()

runtime_results_wide['All_Ranks'] = runtime_results_wide.apply(find_runtime_ranking, axis=1)
methods = results['Method'].unique()
runtime_rank_matrix = pd.DataFrame(
    runtime_results_wide['All_Ranks'].tolist(),
    columns=methods,
    index=runtime_results_wide.index
)
runtime_results_summary = pd.DataFrame({
    'Method': methods,
    'Average_Rank': runtime_rank_matrix.mean().round(4),
    'Median_Rank': runtime_rank_matrix.median().round(4),
    'Std_Rank': runtime_rank_matrix.std().round(4),
    'Times_Ranked_1st': (runtime_rank_matrix == 1).sum(),
    'Times_Ranked_Last': (runtime_rank_matrix == len(methods)).sum()
}).sort_values('Average_Rank')

for method in runtime_results['Method'].unique():
    runtime_results_wide[method] = (
        runtime_results_wide['Mean_Runtime'][method].map('{:.5g}'.format) + 
        '+/e' + runtime_results_wide['Std_Runtime'][method].map('{:.5g}'.format)
    )

runtime_results_wide = runtime_results_wide[
    [method for method in runtime_results['Method'].unique()]
]
runtime_results_wide = runtime_results_wide.reset_index()

# Merge the summaries
separator1 = pd.DataFrame([['---'] * len(results_summary.columns)], 
                         columns=results_summary.columns,
                         index=['MAP PROB RESULTS →'])
separator2 = pd.DataFrame([['---'] * len(results_summary.columns)], 
                         columns=results_summary.columns,
                         index=['RUNTIME RESULTS →'])
summary = pd.concat([
    separator1,
    results_summary,
    separator2,
    runtime_results_summary
])

# Save the results to csvs
print(summary)
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

# Table to send to David, comment out if running pipeline

# all_results.drop(
#     columns = ['Date', 'Query Proportion', 'Evid Proportion', 'MAP Estimate'], 
#     inplace = True
# )
# all_results['Query_Index'] = all_results.groupby(
#     ['Dataset', 'Method']).cumcount() + 1
# all_results = all_results.reset_index()
# all_results.drop(columns=['index', 'Query'], inplace=True)
# all_results.to_csv('results/small_datasets_results_for_david.csv', index=False)
