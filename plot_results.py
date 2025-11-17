import pandas as pd
import numpy as np

all_results = pd.read_csv('benchmark_results.csv')
all_results = all_results[
    all_results["Date"].isin(["2025-11-05 20-02-33"])
]
results = all_results.groupby(
    ['Dataset', 'Method'])['MAP Probability'].agg(
    Mean_MAP_Probability='mean',
    Std_MAP_Probability='std'
).reset_index()
# results.to_csv('benchmark_results_aggregated.csv', index=False)

results_wide = results.pivot(
    index='Dataset', 
    columns='Method', 
    values=['Mean_MAP_Probability', 'Std_MAP_Probability']
)

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

print(results_wide)
results_wide.to_csv('results/small_datasets_results_0.1q0.1e.csv', index=False)

runtime_results = all_results.groupby(['Dataset', 'Method'])['Runtime'].agg(
    Mean_Runtime= 'mean',
    Std_Runtime= 'std'
).reset_index()

runtime_results_wide = runtime_results.pivot(
    index='Dataset',
    columns='Method',
    values=['Mean_Runtime', 'Std_Runtime']
)

for method in runtime_results['Method'].unique():
    runtime_results_wide[method] = (
        runtime_results_wide['Mean_Runtime'][method].map('{:.3e}'.format) + 
        '+/e' + runtime_results_wide['Std_Runtime'][method].map('{:.3e}'.format)
    )

runtime_results_wide = runtime_results_wide[
    [method for method in runtime_results['Method'].unique()]
]
runtime_results_wide = runtime_results_wide.reset_index()