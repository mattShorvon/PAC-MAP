import numpy as np
from spn.node.base import SPN
from spn.node.sum import SumNode
from spn.node.product import ProductNode
from spn.node.gaussian import GaussianNode
from spn.node.multinomial import MultinomialNode
from spn.node.indicator import Indicator
from spn.utils.evidence import Evidence
import copy
from joblib import Parallel, delayed
import os

def sample_parallel(spn: SPN,
                    num_samples: int = 1,
                    evidence: Evidence = None,
                    marginalized: list = None,
                    n_jobs: int = -2):
    # Determine number of workers
    if n_jobs == -1:
        n_workers = os.cpu_count()
    elif n_jobs == -2:
        n_workers = os.cpu_count() - 1
    elif n_jobs < 0:
        n_workers = os.cpu_count() + 1 + n_jobs
    else:
        n_workers = n_jobs
    n_workers = max(1, n_workers)

    # Calculate samples per worker
    samples_per_worker = num_samples // n_workers
    remainder = num_samples % n_workers

    # Create batch sizes (distribute remainder)
    batch_sizes = [samples_per_worker] * n_workers
    for i in range(remainder):
        batch_sizes[i] += 1

    # Sample in parallel
    samples = Parallel(n_jobs=n_workers)(
        delayed(sample)(spn, batch_size, evidence, marginalized)
        for batch_size in batch_sizes
    )

    return [s for batch in samples for s in batch]


def sample(spn: SPN, 
           num_samples: int = 1, 
           evidence: Evidence = None, 
           marginalized: list = None):
    """
    Sample from an SPN.
    
    Args:
        node: Root node of the SPN
        num_samples: Number of samples to generate
        evidence: Evidence dictionary (optional)
    
    Returns:
        List of Evidence objects, one per sample
    """
    if marginalized is None:
        marginalized = []

    samples = []
    for _ in range(num_samples):
        sample = _sample_recursive(spn, 
                                   copy.deepcopy(evidence) or Evidence(),
                                   marginalized)
        samples.append(sample)
    return samples


def _sample_recursive(spn: SPN, 
                      evidence: Evidence = None, 
                      marginalized: list = None):
    """
    Recursively sample from a node given evidence.
    
    Args:
        node: Current node
        evidence: Current evidence (may be partial)
    
    Returns:
        Evidence object with sampled values
    """
    if marginalized is None:
        marginalized = []

    if isinstance(spn, SumNode): # try, but might replace with spn.type == 'sum'
        return _sample_sum(spn, evidence, marginalized)
    elif isinstance(spn, ProductNode):
        return _sample_product(spn, evidence, marginalized)
    elif isinstance(spn, Indicator):
        return _sample_indicator(spn, evidence, marginalized)
    elif isinstance(spn, GaussianNode):
        return _sample_gaussian(spn, evidence, marginalized)
    elif isinstance(spn, MultinomialNode):
        return _sample_multinomial(spn, evidence, marginalized)
    else:
        raise ValueError(f"Unsupported node type: {type(spn)}")


def _sample_sum(node: SPN, 
                evidence: Evidence = None,
                marginalized: list = None):
    """Sample from a sum node by choosing a child based on weights."""
    # Normalize weights
    weights = np.array(node.weights)
    weights = weights / weights.sum()
    
    # Choose a child based on weights
    child_idx = np.random.choice(len(node.children), p=weights)
    child = node.children[child_idx]
    
    # Sample from chosen child
    return _sample_recursive(child, evidence, marginalized)


def _sample_product(node: SPN, 
                    evidence: Evidence = None,
                    marginalized: list = None):
    """Sample from a product node by sampling from all children."""
    current_evidence = Evidence(evidence)
    
    for child in node.children:
        # Sample from each child and merge into evidence.
        # Each child should cover a different variable/set of variables because
        # the product node should be decomposable.
        child_sample = _sample_recursive(child, current_evidence, marginalized)
        current_evidence.merge(child_sample)
    
    return current_evidence


def _sample_indicator(node: SPN, 
                      evidence: Evidence = None,
                      marginalized: list = None):
    """Sample from an indicator (categorical) node."""
    var = node.variable
    
    # If variable already has evidence, use it
    if var in evidence:
        return evidence
    
    # If the variable should be marginalized, skip sampling
    if var in marginalized:
        return evidence
    
    # Otherwise, sample the value this indicator represents
    result = Evidence(evidence)
    result[var] = [node.assignment]
    return result


def _sample_gaussian(node: SPN, 
                     evidence: Evidence = None,
                     marginalized: list = None):
    """Sample from a Gaussian node."""
    var = node.var
    
    # If variable already has evidence, use it
    if var in evidence:
        return evidence

    # If the variable should be marginalized, skip sampling
    if var in marginalized:
        return evidence
    
    # Sample from Gaussian distribution
    value = np.random.normal(node.mean, np.sqrt(node.variance))
    
    result = Evidence(evidence)
    result[var] = [value]
    return result


def _sample_multinomial(node: SPN, 
                        evidence: Evidence = None,
                        marginalized: list = None):
    """Sample from a multinomial node."""
    var = node.var
    
    # If variable already has evidence, use it
    if var in evidence:
        return evidence

    # If the variable should be marginalized, skip sampling
    if var in marginalized:
        return evidence
    
    # Sample from categorical distribution
    probs = np.array(node.probs)
    probs = probs / probs.sum()  # Normalize
    value = np.random.choice(len(probs), p=probs)
    
    result = Evidence(evidence)
    result[var] = [value]
    return result


def sample_with_evidence(node: SPN, 
                         num_samples=1, 
                         evidence: Evidence=None,
                         marginalized: list=None):
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
    if marginalized is None:
        marginalized = []
    
    samples = []
    for _ in range(num_samples):
        sample = _sample_recursive(node, Evidence(evidence), marginalized)
        samples.append(sample)
    
    return samples