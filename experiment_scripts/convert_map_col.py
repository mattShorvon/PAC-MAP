import pandas as pd
import os
import ast


# MAP Estimate columns were recorded differently for ITSELF experiments
# This script just fixes that
all_results = pd.read_csv('../benchmark_results.csv')
all_results = all_results[all_results['Method'] == 'ITSELF']

def _safe_ast(s):
    if pd.isna(s):
        return None
    try:
        return ast.literal_eval(str(s))
    except Exception:
        return s

query_list = all_results['Query'].apply(_safe_ast).tolist()
mapest_list = all_results['MAP Estimate'].apply(_safe_ast).tolist()
convert_list = []
for query, mapest in zip(query_list, mapest_list):
    convert_dict = {}
    for q, val in zip(query, mapest):
        convert_dict[q] = [val]
    convert_list.append(convert_dict)
all_results['MAP Estimate'] = pd.Series(convert_list, index=all_results.index)
all_results.to_csv('../benchmark_results.csv', mode='a', header=False, index=False)