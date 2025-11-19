import numpy as np
from spn.node.base import SPN
from spn.node.sum import SumNode
from spn.node.product import ProductNode
from spn.node.gaussian import GaussianNode
from spn.node.multinomial import MultinomialNode
from spn.node.indicator import Indicator
from spn.utils.evidence import Evidence


def sample(spn: SPN, num_samples: int = 1, evidence: Evidence = None):
    """
    Sample from an SPN.
    
    Args:
        node: Root node of the SPN
        num_samples: Number of samples to generate
        evidence: Evidence dictionary (optional)
    
    Returns:
        List of Evidence objects, one per sample
    """
    samples = []
    for _ in range(num_samples):
        sample = _sample_recursive(spn, evidence or Evidence())
        samples.append(sample)
    return samples


def _sample_recursive(spn: SPN, evidence: Evidence = None):
    """
    Recursively sample from a node given evidence.
    
    Args:
        node: Current node
        evidence: Current evidence (may be partial)
    
    Returns:
        Evidence object with sampled values
    """
    if isinstance(spn, SumNode): # try, but might replace with spn.type == 'sum'
        return _sample_sum(spn, evidence)
    elif isinstance(spn, ProductNode):
        return _sample_product(spn, evidence)
    elif isinstance(spn, Indicator):
        return _sample_indicator(spn, evidence)
    elif isinstance(spn, GaussianNode):
        return _sample_gaussian(spn, evidence)
    elif isinstance(spn, MultinomialNode):
        return _sample_multinomial(spn, evidence)
    else:
        raise ValueError(f"Unsupported node type: {type(spn)}")


def _sample_sum(node: SPN, evidence: Evidence = None):
    """Sample from a sum node by choosing a child based on weights."""
    # Normalize weights
    weights = np.array(node.weights)
    weights = weights / weights.sum()
    
    # Choose a child based on weights
    child_idx = np.random.choice(len(node.children), p=weights)
    child = node.children[child_idx]
    
    # Sample from chosen child
    return _sample_recursive(child, evidence)


def _sample_product(node: SPN, evidence: Evidence = None):
    """Sample from a product node by sampling from all children."""
    current_evidence = Evidence(evidence)
    
    for child in node.children:
        # Sample from each child and merge into evidence.
        # Each child should cover a different variable/set of variables because
        # the product node should be decomposable.
        child_sample = _sample_recursive(child, current_evidence)
        current_evidence.merge(child_sample)
    
    return current_evidence


def _sample_indicator(node: SPN, evidence: Evidence = None):
    """Sample from an indicator (categorical) node."""
    var = node.variable
    
    # If variable already has evidence, use it
    if var in evidence:
        return evidence
    
    # Otherwise, sample the value this indicator represents
    result = Evidence(evidence)
    result[var] = [node.assignment]
    return result


def _sample_gaussian(node: SPN, evidence: Evidence = None):
    """Sample from a Gaussian node."""
    var = node.var
    
    # If variable already has evidence, use it
    if var in evidence:
        return evidence
    
    # Sample from Gaussian distribution
    value = np.random.normal(node.mean, np.sqrt(node.variance))
    
    result = Evidence(evidence)
    result[var] = [value]
    return result


def _sample_multinomial(node: SPN, evidence: Evidence = None):
    """Sample from a multinomial node."""
    var = node.var
    
    # If variable already has evidence, use it
    if var in evidence:
        return evidence
    
    # Sample from categorical distribution
    probs = np.array(node.probs)
    probs = probs / probs.sum()  # Normalize
    value = np.random.choice(len(probs), p=probs)
    
    result = Evidence(evidence)
    result[var] = [value]
    return result


def sample_with_evidence(node: SPN, num_samples=1, evidence: Evidence =None):
    """
    Sample from an SPN with evidence.
    
    Args:
        node: Root node of the SPN
        num_samples: Number of samples to generate
        evidence: Evidence dictionary with fixed values
    
    Returns:
        List of Evidence objects
    """
    if evidence is None:
        evidence = Evidence()
    
    samples = []
    for _ in range(num_samples):
        sample = _sample_recursive(node, Evidence(evidence))
        samples.append(sample)
    
    return samples

if __name__ == "__main__":
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

    conditional_samples = sample_with_evidence(spn, num_samples=10, evidence=evidence)

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