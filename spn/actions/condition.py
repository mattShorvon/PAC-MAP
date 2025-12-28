import numpy as np
from spn.node.base import SPN
from spn.node.sum import SumNode
from spn.node.product import ProductNode
from spn.node.gaussian import GaussianNode
from spn.node.multinomial import MultinomialNode
from spn.node.indicator import Indicator
from spn.utils.evidence import Evidence
from spn.structs import Variable
from typing import List
import copy


def condition_spn(spn: SPN, 
                  evidence: Evidence,
                  marginalized: List[Variable] = None) -> SPN:
    """
    Create a new SPN conditioned on evidence by re-weighting sum nodes.
    
    This creates a new SPN structure where:
    - Sum node weights are updated to reflect P(child | evidence)
    - Leaf nodes with evidence are "pruned" (kept for structure but marked)
    - The resulting SPN represents P(Q | E) directly
    
    Args:
        spn: Original SPN
        evidence: Evidence to condition on
        marginalized: Variables to marginalize (not condition on)
    
    Returns:
        New conditioned SPN that can be sampled without passing evidence
    """
    if marginalized is None:
        marginalized = []
    
    if evidence is None or len(evidence) == 0:
        return copy.deepcopy(spn)
    
    # First pass: compute likelihoods bottom-up
    node_likelihoods = _compute_likelihoods(spn, evidence, marginalized)
    
    # Second pass: create new conditioned SPN top-down
    visited = {}
    conditioned_root = _condition_recursive(spn, evidence, marginalized, 
                                           node_likelihoods, visited)
    
    return conditioned_root


def _compute_likelihoods(node: SPN,
                        evidence: Evidence,
                        marginalized: List[Variable]) -> dict:
    """Compute P(evidence | subtree) for each node."""
    likelihoods = {}
    
    def compute(n):
        if id(n) in likelihoods:
            return likelihoods[id(n)]
        
        if isinstance(n, SumNode):
            child_liks = [compute(child) for child in n.children]
            # P(e) = sum_i w_i * P_i(e)
            lik = sum(w * cl for w, cl in zip(n.weights, child_liks))
            
        elif isinstance(n, ProductNode):
            child_liks = [compute(child) for child in n.children]
            # P(e) = prod_i P_i(e)
            lik = np.prod(child_liks)
            
        elif isinstance(n, Indicator):
            var = n.variable
            if var in evidence and var not in marginalized:
                # Check if evidence matches
                lik = 1.0 if evidence[var][0] == n.assignment else 0.0
            else:
                lik = 1.0
                
        elif isinstance(n, GaussianNode):
            var = n.var
            if var in evidence and var not in marginalized:
                value = evidence[var][0]
                # Gaussian likelihood
                lik = (1.0 / np.sqrt(2 * np.pi * n.variance)) * \
                      np.exp(-0.5 * ((value - n.mean) ** 2) / n.variance)
            else:
                lik = 1.0
                
        elif isinstance(n, MultinomialNode):
            var = n.var
            if var in evidence and var not in marginalized:
                value = int(evidence[var][0])
                probs = np.array(n.probs)
                probs = probs / probs.sum()
                lik = probs[value] if 0 <= value < len(probs) else 0.0
            else:
                lik = 1.0
        else:
            raise ValueError(f"Unsupported node type: {type(n)}")
        
        likelihoods[id(n)] = lik
        return lik
    
    compute(node)
    return likelihoods


def _condition_recursive(node: SPN,
                        evidence: Evidence,
                        marginalized: List[Variable],
                        node_likelihoods: dict,
                        visited: dict) -> SPN:
    """
    Create conditioned copy of node with re-weighted sum nodes.
    Uses memoization to handle shared subgraphs.
    """
    if id(node) in visited:
        return visited[id(node)]
    
    if isinstance(node, SumNode):
        # Create new sum node with re-weighted children
        new_children = []
        new_weights = []
        
        for child, weight in zip(node.children, node.weights):
            # Recursively condition child
            new_child = _condition_recursive(child, evidence, marginalized,
                                            node_likelihoods, visited)
            new_children.append(new_child)
            
            # Re-weight based on evidence
            # P(child | e) ∝ w * P(e | child)
            child_lik = node_likelihoods[id(child)]
            new_weight = weight * child_lik
            new_weights.append(new_weight)
        
        # Normalize weights
        total_weight = sum(new_weights)
        if total_weight > 0:
            new_weights = [w / total_weight for w in new_weights]
        else:
            # All children have zero likelihood (shouldn't happen with valid evidence)
            new_weights = [1.0 / len(new_weights)] * len(new_weights)
        
        # Create new sum node
        new_node = SumNode()
        new_node.children = new_children
        new_node.weights = new_weights
        
    elif isinstance(node, ProductNode):
        # Create new product node with conditioned children
        new_children = []
        for child in node.children:
            new_child = _condition_recursive(child, evidence, marginalized,
                                            node_likelihoods, visited)
            new_children.append(new_child)
        
        new_node = ProductNode()
        new_node.children = new_children
        
    elif isinstance(node, Indicator):
        # Keep indicators as-is (evidence will be handled during sampling)
        # Deep copy to avoid modifying original
        new_node = Indicator(node.variable, node.assignment)
        
    elif isinstance(node, GaussianNode):
        # Keep Gaussian nodes as-is
        new_node = GaussianNode(node.var, node.mean, node.variance)
        
    elif isinstance(node, MultinomialNode):
        # Keep Multinomial nodes as-is
        new_node = MultinomialNode(node.var, node.probs.copy())
        
    else:
        raise ValueError(f"Unsupported node type: {type(node)}")
    
    visited[id(node)] = new_node
    return new_node