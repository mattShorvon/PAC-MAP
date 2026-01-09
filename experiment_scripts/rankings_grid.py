import pandas as pd
import numpy as np
import argparse
from pathlib import Path
from spn.io.file import from_file
from datetime import datetime

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

results = all_results.groupby(
    ['Dataset', 'Method', 'Experiment ID'])['MAP Probability'].agg(
    Mean_MAP_Probability='mean',
    Std_MAP_Probability='std'
).reset_index()
# results.to_csv('benchmark_results_aggregated.csv', index=False)

# Convert prob results to wide format
results_wide = results.pivot(
    index=['Dataset', 'Method'], 
    columns='Experiment ID', 
    values=['Mean_MAP_Probability']  
)

# Group by Dataset and Experiment ID, collect method ranks
method_order = sorted(results['Method'].unique())
rankings_per_cell = []
for dataset in results['Dataset'].unique():
    row_data = {'Dataset': dataset}
    
    for exp_id in exp_ids:
        # Get all methods for this dataset and experiment
        subset = results[
            (results['Dataset'] == dataset) & 
            (results['Experiment ID'] == exp_id)
        ].copy()
        
        if subset.empty:
            row_data[exp_id] = None
            continue
        
        # Calculate ranks
        log_probs = np.log(subset['Mean_MAP_Probability'] + 1e-300)
        subset['Rank'] = log_probs.rank(method='min', ascending=False)
        
        # Create ordered list of ranks
        subset_indexed = subset.set_index('Method')
        ranks_list = [
            subset_indexed.loc[method, 'Rank'] if method in subset_indexed.index else None
            for method in method_order
        ]
        
        row_data[exp_id] = ranks_list
    
    rankings_per_cell.append(row_data)

rankings_df = pd.DataFrame(rankings_per_cell)
rankings_df.columns = ['Dataset', '10% Query', '25% Query', '40% Query']

# Save the results to csvs
print(rankings_df)
datetime_str = datetime.now().strftime("%d-%m-%Y %H-%M-%S")
rankings_df.to_csv(
    Path('results') / f'benchmark_rankings_{datetime_str}.csv'
)