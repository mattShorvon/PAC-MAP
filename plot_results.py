import pandas as pd
import numpy as np

all_results = pd.read_csv('benchmark_results.csv')
all_results = all_results[all_results['Date'] == "2025-10-24 11-58-00"]
results = all_results.groupby(
    ['Dataset', 'Method'])['MAP Probability'].agg(
    Mean_MAP_Probability='mean',
    Std_MAP_Probability='std'
).reset_index()
results.to_csv('benchmark_results_aggregated.csv', index=False)