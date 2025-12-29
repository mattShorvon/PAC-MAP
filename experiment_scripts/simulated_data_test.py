import numpy as np
import pandas as pd
from spn.learn import gens
from pathlib import Path
from spn.data.partitioned_data import PartitionedData
from spn.io.file import to_file, from_file
from spn.actions.sample import sample_multiproc, sample
from spn.actions.likelihood import likelihood_multiproc
from spn.actions.condition import condition_spn
from spn.utils.evidence import Evidence
import tempfile


# Script that creates a toy dataset with a known data-generating process, so 
# that we have ground truth probabilities, and tests the likelihood() and 
# sampling() functions

if __name__ == "__main__":
    # Initialise variables and create the toy data
    n = 1000
    create_new_data = False
    learn = False
    test_likelihood = False
    test_sampling = False
    vals = [0, 1]
    p_x1 = 0.7
    path = "test_inputs/toy_data/toy_data.train.data"
    csv_path = "test_inputs/toy_data/toy_data.csv"
    if create_new_data:
        X1 = np.random.choice(vals, n, p=[1 - p_x1, p_x1])
        X2 = np.array([
            np.random.choice(vals, 1, p=[1 - 0.8, 0.8])[0] 
            if x == 1 
            else np.random.choice(vals, 1, p=[1 - 0.2, 0.2])[0] for x in X1
        ])
        X3 = np.array([
            np.random.choice(vals, 1, p=[1 - 0.2, 0.2])[0] 
            if x == 1 
            else np.random.choice(vals, 1, p=[1 - 0.8, 0.8])[0] for x in X1
        ])
        data = pd.DataFrame({
            'X1': X1,
            'X2': X2,
            'X3': X3
        })

        # Save the toy data
        data.to_csv(csv_path, index=False)
        with open(path, 'w') as file:
            for i, var in enumerate(data.columns.tolist()):
                n_categories = data.nunique()[var]
                file.write(' '.join(['var', str(i), str(n_categories), '\n']))
            for _, row in data.iterrows():
                line = ' '.join([str(i) for i in row.to_list()])
                file.write(line + '\n')

        print('File written')
    else:
        data = pd.read_csv(csv_path)
        print('Data loaded')
    print("\n=== Ground Truth from Generated Data ===")

    # Total counts
    n_total = len(data)
    n_x1_1 = len(data[data["X1"] == 1])
    n_x1_0 = len(data[data["X1"] == 0])

    print(f"Total samples: {n_total}")
    print(f"X1=1: {n_x1_1} ({n_x1_1/n_total:.3f})")
    print(f"X1=0: {n_x1_0} ({n_x1_0/n_total:.3f})")

    # P(X2 | X1)
    n_x1_1_x2_1 = len(data[(data["X1"] == 1) & (data["X2"] == 1)])
    n_x1_1_x2_0 = len(data[(data["X1"] == 1) & (data["X2"] == 0)])
    n_x1_0_x2_1 = len(data[(data["X1"] == 0) & (data["X2"] == 1)])
    n_x1_0_x2_0 = len(data[(data["X1"] == 0) & (data["X2"] == 0)])

    print(f"\nP(X2=1 | X1=1) = {n_x1_1_x2_1/n_x1_1:.3f} (expected: 0.800)")
    print(f"P(X2=0 | X1=1) = {n_x1_1_x2_0/n_x1_1:.3f} (expected: 0.200)")
    print(f"P(X2=1 | X1=0) = {n_x1_0_x2_1/n_x1_0:.3f} (expected: 0.200)")
    print(f"P(X2=0 | X1=0) = {n_x1_0_x2_0/n_x1_0:.3f} (expected: 0.800)")

    # P(X3 | X1)
    n_x1_1_x3_1 = len(data[(data["X1"] == 1) & (data["X3"] == 1)])
    n_x1_1_x3_0 = len(data[(data["X1"] == 1) & (data["X3"] == 0)])
    n_x1_0_x3_1 = len(data[(data["X1"] == 0) & (data["X3"] == 1)])
    n_x1_0_x3_0 = len(data[(data["X1"] == 0) & (data["X3"] == 0)])

    print(f"\nP(X3=1 | X1=1) = {n_x1_1_x3_1/n_x1_1:.3f} (expected: 0.200)")
    print(f"P(X3=0 | X1=1) = {n_x1_1_x3_0/n_x1_1:.3f} (expected: 0.800)")
    print(f"P(X3=1 | X1=0) = {n_x1_0_x3_1/n_x1_0:.3f} (expected: 0.800)")
    print(f"P(X3=0 | X1=0) = {n_x1_0_x3_0/n_x1_0:.3f} (expected: 0.200)")
    

    # Learn an SPN on the toy data
    spn_path = Path('test_inputs/toy_data/toy_data.spn')
    train_data = PartitionedData(Path(path), 1.0)
    if learn:
        spn = gens(train_data.scope, train_data.training_data, kclusters=3, pval=0.5, 
                numeric=False)
        to_file(spn=spn, file_path=spn_path)
        print('SPN learned and saved')
    else:
        spn = from_file(spn_path)
        print('SPN loaded')


    # Test the likelihood functions:
    var_x1 = train_data.scope[0]
    var_x2 = train_data.scope[1]
    var_x3 = train_data.scope[2]
    if test_likelihood:
        cond_x1_1 = spn.value(Evidence({var_x1: [1]}))
        cond_x1_0 = spn.value(Evidence({var_x1: [0]}))
        query_1 = Evidence({var_x1: [1], var_x2: [1]})
        print(f"P(X2 = 1 | X1 = 1) according to spn.value(): {spn.value(query_1) / cond_x1_1}")
        query_2 = Evidence({var_x1: [1], var_x2: [0]})
        print(f"P(X2 = 0 | X1 = 1) according to spn.value(): {spn.value(query_2) / cond_x1_1}")
        query_3 = Evidence({var_x1: [0], var_x2: [1]})
        print(f"P(X2 = 1 | X1 = 0) according to spn.value(): {spn.value(query_3) / cond_x1_0}")
        query_4 = Evidence({var_x1: [0], var_x2: [0]})
        print(f"P(X2 = 0 | X1 = 0) according to spn.value(): {spn.value(query_4) / cond_x1_0}")
        query_1 = Evidence({var_x1: [1], var_x3: [1]})
        print(f"P(X3 = 1 | X1 = 1) according to spn.value(): {spn.value(query_1) / cond_x1_1}")
        query_2 = Evidence({var_x1: [1], var_x3: [0]})
        print(f"P(X3 = 0 | X1 = 1) according to spn.value(): {spn.value(query_2) / cond_x1_1}")
        query_3 = Evidence({var_x1: [0], var_x3: [1]})
        print(f"P(X3 = 1 | X1 = 0) according to spn.value(): {spn.value(query_3) / cond_x1_0}")
        query_4 = Evidence({var_x1: [0], var_x3: [0]})
        print(f"P(X3 = 0 | X1 = 0) according to spn.value(): {spn.value(query_4) / cond_x1_0}")

        cond_x1_1, cond_x1_0 = np.exp(likelihood_multiproc(
            spn_path, 
            [Evidence({var_x1: [1]}), Evidence({var_x1: [0]})], 
            n_jobs = 2
        ))
        query_1 = Evidence({var_x1: [1], var_x2: [1]})
        query_2 = Evidence({var_x1: [1], var_x2: [0]})
        query_3 = Evidence({var_x1: [0], var_x2: [1]})
        query_4 = Evidence({var_x1: [0], var_x2: [0]})
        query_5 = Evidence({var_x1: [1], var_x3: [1]})
        query_6 = Evidence({var_x1: [1], var_x3: [0]})
        query_7 = Evidence({var_x1: [0], var_x3: [1]})
        query_8 = Evidence({var_x1: [0], var_x3: [0]})
        ans1, ans2, ans5, ans6 = np.exp(likelihood_multiproc(
            spn_path,
            [query_1, query_2, query_5, query_6],
            n_jobs=2
        )) / cond_x1_1
        ans3, ans4, ans7, ans8 = np.exp(likelihood_multiproc(
            spn_path,
            [query_3, query_4, query_7, query_8],
            n_jobs=2
        )) / cond_x1_0
        print(f"P(X2 = 1 | X1 = 1) according to likelihood(): {ans1}")
        print(f"P(X2 = 0 | X1 = 1) according to likelihood(): {ans2}")
        print(f"P(X2 = 1 | X1 = 0) according to likelihood(): {ans3}")
        print(f"P(X2 = 0 | X1 = 0) according to likelihood(): {ans4}")
        print(f"P(X3 = 1 | X1 = 1) according to likelihood(): {ans5}")
        print(f"P(X3 = 0 | X1 = 1) according to likelihood(): {ans6}")
        print(f"P(X3 = 1 | X1 = 0) according to likelihood(): {ans7}")
        print(f"P(X3 = 0 | X1 = 0) according to likelihood(): {ans8}")

    # Testing the sampling
    if test_sampling:
        n = 10000
        query_1 = Evidence({var_x1: [1]})
        samples_x1_1_x2 = sample_multiproc(spn_path, num_samples=n, evidence=query_1, 
                                        marginalized=[var_x3])
        count_x2_1 = sum([1 if sample[var_x2] == [1] else 0 for sample in samples_x1_1_x2])
        print(f"P(X2 = 1 | X1 = 1) according to sampling(): {count_x2_1 / len(samples_x1_1_x2)}")
        print(f"P(X2 = 0 | X1 = 1) according to sampling(): {(n - count_x2_1) / len(samples_x1_1_x2)}")

        samples_x1_1_x3 = sample_multiproc(spn_path, num_samples=n, evidence=query_1, 
                                        marginalized=[var_x2])
        count_x3_1 = sum([1 if sample[var_x3] == [1] else 0 for sample in samples_x1_1_x3])
        print(f"P(X3 = 1 | X1 = 1) according to sampling(): {count_x3_1 / len(samples_x1_1_x3)}")
        print(f"P(X3 = 0 | X1 = 1) according to sampling(): {(n - count_x3_1) / len(samples_x1_1_x3)}")

        query_0 = Evidence({var_x1: [0]})
        samples_x1_0_x2 = sample_multiproc(spn_path, num_samples=n, evidence=query_0,
                                        marginalized=[var_x3])
        count_x2_1 = sum([1 if sample[var_x2] == [1] else 0 for sample in samples_x1_0_x2])
        print(f"P(X2 = 1 | X1 = 0) according to sampling(): {count_x2_1 / len(samples_x1_0_x2)}")
        print(f"P(X2 = 0 | X1 = 0) according to sampling(): {(n - count_x2_1) / len(samples_x1_0_x2)}")

        samples_x1_0_x3 = sample_multiproc(spn_path, num_samples=n, evidence=query_0,
                                        marginalized=[var_x2])
        count_x3_1 = sum([1 if sample[var_x3] == [1] else 0 for sample in samples_x1_0_x3])
        print(f"P(X3 = 1 | X1 = 0) according to sampling(): {count_x3_1 / len(samples_x1_0_x3)}")
        print(f"P(X3 = 0 | X1 = 0) according to sampling(): {(n - count_x3_1) / len(samples_x1_0_x3)}")

    # Debugging the sampling
    query_0 = Evidence({var_x1: [0]})
    spn_conditioned = condition_spn(spn, query_0)
    with tempfile.NamedTemporaryFile(mode='wb', suffix='.spn', delete=False) as f:
        conditioned_spn_path = Path(f.name)
    to_file(spn_conditioned, conditioned_spn_path)

    n = 10000
    samples_x1_0_x2 = sample_multiproc(conditioned_spn_path, num_samples=n, evidence=None,
                                       marginalized=[var_x3])
    count_x2_1 = sum([1 if sample[var_x2] == [1] else 0 for sample in samples_x1_0_x2])
    print(f"P(X2 = 1 | X1 = 0) according to sampling(): {count_x2_1 / len(samples_x1_0_x2)}")
    print(f"P(X2 = 0 | X1 = 0) according to sampling(): {(n - count_x2_1) / len(samples_x1_0_x2)}")

    samples_x1_0_x3 = sample_multiproc(conditioned_spn_path, num_samples=n, evidence=None,
                                       marginalized=[var_x2])
    count_x3_1 = sum([1 if sample[var_x3] == [1] else 0 for sample in samples_x1_0_x3])
    print(f"P(X3 = 1 | X1 = 0) according to sampling(): {count_x3_1 / len(samples_x1_0_x3)}")
    print(f"P(X3 = 0 | X1 = 0) according to sampling(): {(n - count_x3_1) / len(samples_x1_0_x3)}")

    conditioned_spn_path.unlink()

    # sample_test1 = sample(spn, num_samples=1, evidence=query_0)
    # sample_test2 = sample(spn, num_samples=1, evidence=query_0,
    #                       marginalized=[var_x3])