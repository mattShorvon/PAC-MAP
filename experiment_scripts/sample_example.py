from spn.io.file import from_file
from spn.actions.sample import sample, sample_with_evidence
from spn.utils.evidence import Evidence
from pathlib import Path

# Load an SPN
spn = from_file(Path("test_inputs/iris/iris.spn"))

# Example 1: Sample without evidence
samples = sample(spn, num_samples=10)

for i, s in enumerate(samples):
    print(f"Sample {i}: {dict(s)}")

# Example 2: Sample with evidence
evidence = Evidence()
evidence[spn.scope()[0]] = [1]  # Fix variable 0 to value 1
evidence[spn.scope()[1]] = [0]  # Fix variable 1 to value 0

conditional_samples = sample_with_evidence(spn, num_samples=1, evidence=evidence)

for i, s in enumerate(conditional_samples):
    print(f"Conditional sample {i}: {dict(s)}")

# Example 3: Convert samples to numpy array
import numpy as np

samples = sample(spn, num_samples=100)
num_vars = len(spn.scope())

# Create matrix
sample_matrix = np.zeros((len(samples), num_vars))
for i, s in enumerate(samples):
    for var, value in s.items():
        sample_matrix[i, var.id] = value[0]

print(f"Sample matrix shape: {sample_matrix.shape}")
print(sample_matrix[:5])  # First 5 samples