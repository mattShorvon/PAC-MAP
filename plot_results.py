import pandas as pd
import numpy as np
import argparse

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

# Find the winning method for each row (method that produced the MAP 
# assignment with the highest probability)
def find_winners(row):
    mean_probs = row['Mean_MAP_Probability']
    max_prob = mean_probs.max()
    winners = mean_probs[mean_probs == max_prob].index.tolist()
    return winners

results_wide['Winner'] = results_wide.apply(find_winners, axis=1)

# Count the number of times each method was a winner
winner_counts = results_wide['Winner'].explode().value_counts()
for method in results['Method'].unique(): 
    if method not in winner_counts:
        winner_counts[method] = 0

# Combine mean and std into single strings
for method in results['Method'].unique():
    results_wide[method] = (
        results_wide['Mean_MAP_Probability'][method].map('{:.3e}'.format) + 
        ' +/- ' + 
        results_wide['Std_MAP_Probability'][method].map('{:.3e}'.format)
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

def find_runtime_winners(row):
    mean_probs = row['Mean_Runtime']
    max_prob = mean_probs.min()
    winners = mean_probs[mean_probs == max_prob].index.tolist()
    return winners

runtime_results_wide['Winner'] = runtime_results_wide.apply(
    find_runtime_winners, axis=1
)
winners_runtime = runtime_results_wide['Winner'].explode().value_counts()
for method in results['Method'].unique(): 
    if method not in winners_runtime:
        winners_runtime[method] = 0

for method in runtime_results['Method'].unique():
    runtime_results_wide[method] = (
        runtime_results_wide['Mean_Runtime'][method].map('{:.3e}'.format) + 
        '+/e' + runtime_results_wide['Std_Runtime'][method].map('{:.3e}'.format)
    )

runtime_results_wide = runtime_results_wide[
    [method for method in runtime_results['Method'].unique()]
]
runtime_results_wide = runtime_results_wide.reset_index()

# Merge summaries into one
winner_summary = pd.DataFrame({
    'Method': winner_counts.index.union(winners_runtime.index),
})
winner_summary['Probability_Wins'] = winner_summary['Method'].map(winner_counts).fillna(0).astype(int)
winner_summary['Runtime_Wins'] = winner_summary['Method'].map(winners_runtime).fillna(0).astype(int)
print("\nWinner Summary:")
print(winner_summary)

# Save the results to csvs
print(results_wide)
results_wide.to_csv(
    f'results/{dataset_name}_results_{q_percent}q{e_percent}e_{datetime_str}.csv', 
    index=False
)
runtime_results_wide.to_csv(
    f'results/{dataset_name}_runtimes_{q_percent}q{e_percent}e_{datetime_str}.csv', 
    index=False
)
winner_summary.to_csv(
    f'results/{dataset_name}_summary_{q_percent}q{e_percent}e_{datetime_str}.csv',
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
