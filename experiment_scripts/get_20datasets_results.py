import pandas as pd

# Extract the 20-datasets results that you want to present for the IJCAI
# rebuttal and save them to the bottom of the csv with the correct experiment id

df = pd.read_csv('../benchmark_results.csv')
methods = ['PAC_MAP_Hamming', 'ITSELF', 'Hybrid Belief-Propagation']
datasets = ['jester', 'connect4', 'bnetflix', 'kosarek', 'dna', 'pumsb_star']
experiments = ["29_12_25_1", "29_12_25_2", "06_01_26_3"]
mask = (
    (df['Method'].isin(methods)) &
    (df['Dataset']).isin(datasets) &
    (df['Experiment ID']).isin(experiments)
)
results = df.loc[mask, :].copy()
results = results.sort_values(
    ['Experiment ID', 'Dataset', 'Query_ID', 'Method']
).reset_index(drop=True)
results.to_csv('../benchmark_results_ijcaireview.csv', index=False)