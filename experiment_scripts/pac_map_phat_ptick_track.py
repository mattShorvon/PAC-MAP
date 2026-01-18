import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from spn.io.file import from_file
from spn.actions.map_algorithms.pac_map import pac_map_tracking
from spn.utils.evidence import Evidence

# Load SPN
dataset = "kdd"
spn_path = Path(f"20-datasets/{dataset}/{dataset}.spn")
spn = from_file(spn_path)

# Define evidence
queries, evidences = [], []
with open(f"20-datasets/{dataset}/{dataset}_0.25q_0.75e.map") as f:
    for line_no, line in enumerate(f):
        if line_no % 2 == 0: 
            query = [spn.scope()[int(var_id)] for var_id in line.split()]
            queries.append(query)
        else: 
            evid_info = line.split()
            index = 0
            evidence = Evidence()
            while index < len(evid_info):
                var_id = int(evid_info[index])
                val = int(evid_info[index + 1])
                evidence[spn.scope()[var_id]] = [val]
                index += 2
            evidences.append(evidence)
evidence = evidences[0]

# Run PAC-MAP with tracking
print("Loaded spn and evidence, running pac-map")
q_hat, p_hat, tracking_df = pac_map_tracking(
    spn=spn,
    spn_path=spn_path,
    evidence=evidence,
    batch_size=100,
    err_tol=0.01,
    fail_prob=0.0001,
    sample_cap=50000,
    save_tracking=False,
    tracking_path=Path("results/pac_map_tracking.csv")
)

print(f"Best estimate: {q_hat}")
print(f"Best probability: {p_hat}")
print(f"\nTracking data:\n{tracking_df}")

# Plot convergence
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Plot p_hat progression
axes[0, 0].plot(tracking_df['m'], tracking_df['p_hat'], 'b-', linewidth=2)
axes[0, 0].set_xlabel('Samples (m)')
axes[0, 0].set_ylabel('p_hat')
axes[0, 0].set_title('Best Probability Estimate Over Time')
axes[0, 0].grid(True, alpha=0.3)

# Plot p_tick progression
axes[0, 1].plot(tracking_df['m'], tracking_df['p_tick'], 'r-', linewidth=2)
axes[0, 1].set_xlabel('Samples (m)')
axes[0, 1].set_ylabel('p_tick')
axes[0, 1].set_title('Remaining Probability Mass Over Time')
axes[0, 1].grid(True, alpha=0.3)

# Plot M progression
axes[1, 0].plot(tracking_df['m'], tracking_df['M'], 'g-', linewidth=2)
axes[1, 0].set_xlabel('Samples (m)')
axes[1, 0].set_ylabel('M (Required Samples)')
axes[1, 0].set_title('Sample Requirement Over Time')
axes[1, 0].set_yscale('log')
axes[1, 0].grid(True, alpha=0.3)

# Plot p_hat and p_tick together
axes[1, 1].plot(tracking_df['m'], tracking_df['p_hat'], 'b-', linewidth=2, label='p_hat')
axes[1, 1].plot(tracking_df['m'], tracking_df['p_tick'], 'r-', linewidth=2, label='p_tick')
axes[1, 1].set_xlabel('Samples (m)')
axes[1, 1].set_ylabel('Probability')
axes[1, 1].set_title('p_hat vs p_tick')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
plt.savefig('results/pac_map_convergence.pdf', dpi=300)