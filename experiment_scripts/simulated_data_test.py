import numpy as np
import pandas as pd
from spn.learn import gens
from pathlib import Path
from spn.data.partitioned_data import PartitionedData


# Script that creates a toy dataset with a known data-generating process, so 
# that we have ground truth probabilities
vals = [0, 1]
n = 1000
p_x1 = 0.7
X1 = np.random.choice(vals, n, p=[1 - p_x1, p_x1])
X2 = np.array([
    np.random.choice(vals, 1, p=[1 - 0.7, 0.7])[0] 
    if x == 1 
    else np.random.choice(vals, 1, p=[1 - 0.3, 0.3])[0] for x in X1
])
X3 = np.array([
    np.random.choice(vals, 1, p=[1 - 0.4, 0.4])[0] 
    if x == 1 
    else np.random.choice(vals, 1, p=[1 - 0.6, 0.6])[0] for x in X1
])
data = pd.DataFrame({
    'X1': X1,
    'X2': X2,
    'X3': X3
})

path = "test_inputs/toy_data/toy_data.train.data"
with open(path, 'w') as file:
    for i, var in enumerate(data.columns.tolist()):
        n_categories = data.nunique()[var]
        file.write(' '.join(['var', str(i), str(n_categories), '\n']))
    for _, row in data.iterrows():
        line = ' '.join([str(i) for i in row.to_list()])
        file.write(line + '\n')

print('File written')

train_data = PartitionedData(Path(path), 1.0)
spn = gens(train_data.scope(), train_data.data, kclusters=3, pval=0.5, 
           numeric=False)
print('SPN learned')