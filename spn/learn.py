"""Learn: Algorithms for learning SPNs from data"""
import logging
from concurrent.futures import ProcessPoolExecutor

from typing import List, Mapping
import numpy
from spn.cluster.kmedoid import kmedoid
from spn.data.parsed_data import Variable
from spn.node.multinomial import MultinomialNode
from spn.node.product import ProductNode
from spn.node.sum import SumNode
from spn.node.gaussian import GaussianNode
from spn.node.indicator import Indicator
from spn.indep.graph import IndependencyGraph
from spn.structs import Vardata


def create_gaussian_node(
    variable: Variable, data: numpy.array, var_id_to_array_index: Mapping[int, int]
):
    """Returns a leaf (Gaussian node) by calculating mean and average from the
    values taken by the variable in this slice of data"""
    logging.info("Creating new Gaussian leaf...")
    values = numpy.array([
        data_sample[var_id_to_array_index[variable.id]]
        for data_sample in data
    ])
    mu = numpy.mean(values)
    sigma = numpy.std(values)
    return GaussianNode(variable, mu, sigma)


def create_multinomial_node(
    variable: Variable, data: numpy.array, var_id_to_array_index: Mapping[int, int]
):
    """Returns a leaf (multinomial node) by counting the values taken by the
    variable in this slice of data"""
    logging.info("Creating new leaf...")
    counts = numpy.zeros(variable.n_categories)
    for data_sample in data:
        counts[data_sample[var_id_to_array_index[variable.id]]] += 1
    return MultinomialNode(variable, counts=counts)


def create_indicators(
    variable: Variable, data: numpy.array, var_id_to_array_index: Mapping[int, int]
) -> Indicator:
    """Returns a sum node with leaves (indicators) by counting the values taken by
    the variable in this slice of data"""
    logging.info("Creating new leaf...")
    counts = numpy.zeros(variable.n_categories)
    for data_sample in data:
        counts[data_sample[var_id_to_array_index[variable.id]]] += 1
    return MultinomialNode(variable, counts=counts).to_sum_node()


def fully_factorized_form(
    scope: List[Variable], data: numpy.array, var_id_to_array_index: Mapping[int, int]
):
    """When all instances all approximately equal (the independence checkings
    returned only one set) it's said that the data is fully factorized, and in an
    SPN a Product Node is created, having leaves with the independent variables. """
    product_node = ProductNode()
    for variable in scope:
        leaf = create_indicators(variable, data, var_id_to_array_index)
        product_node.add_child(leaf)
    return product_node


def detach_independent_vars(
    data: numpy.array, var_id_to_array_index: Mapping[int, int],
    independent_set_of_vars: List[int], var_id_to_var: Mapping[int, Variable],
    kclusters: int, pval: float, numeric: bool
):
    new_training_data_indexes = []
    new_scope = []
    for var_id in independent_set_of_vars:
        new_training_data_indexes.append(var_id_to_array_index[var_id])
        new_scope.append(var_id_to_var[var_id])
    new_training_data = data[:, new_training_data_indexes]
    return gens(new_scope, new_training_data, kclusters, pval, numeric)


def gens(
    scope: List[Variable], data: numpy.array, kclusters: int, pval: float,
    numeric: bool
):
    """The classical algorithm from Poon and Domingos for learning the structure of
    SPNs. Based on the article "Learning the Structure of Sum Product Networks"
    (2013)"""
    if kclusters <= 0:
        message = "Must invoke gens with a positive number of clusters"
        logging.error(message)
        raise ValueError(message)

    # Indexing a numpy array is only possible with natural numbers starting from
    # 0. With this, we can keep using the variable ids in order to have a mapping
    # from id to index, so the first variable can have id 3 and still be indexed
    # in the first column [:, 0] of the data
    var_id_to_array_index = {
        var.id: column for var, column in zip(scope, range(len(data[0])))
    }

    # If the data's scope is unary, we have a univariate distribution, so we
    # return a leaf
    if len(scope) == 1:
        variable = scope[0]  # The only one in the scope
        if numeric:
            return create_gaussian_node(variable, data, var_id_to_array_index)
        return create_indicators(variable, data, var_id_to_array_index)

    # Else  we check for independent subsets of variables, separating them
    # in k partitions.
    # Each partition will be pairwise independent with the others
    logging.info("Creating vardatas for independence test")
    var_datas = []
    for variable in scope:
        # separate the training data for this one variable
        training_data = numpy.array(data[:, var_id_to_array_index[variable.id]])
        var_datas.append(Vardata(variable, training_data))

    logging.info("Creating new Independence Graph")
    independency_graph = IndependencyGraph(var_datas, pval, numeric)

    if len(independency_graph.kset) > 1:
        # With more than 1 kset on the graph, we can partition the sets of
        # variables in the data into disjunct domains in order to form product
        # nodes. This means that we'll need a new set of data samples with only
        # these variables, independent of the others, and a new scope with only
        # these variables
        logging.info("Found independence. Separating independent sets...")
        product_node = ProductNode()
        var_id_to_var = {
            var.id: var for var in scope
        }

        with ProcessPoolExecutor(max_workers=2) as executor:
            children = [
                executor.submit(detach_independent_vars, data, var_id_to_array_index,
                independent_set_of_vars, var_id_to_var, kclusters, pval,
                numeric).result()
                for independent_set_of_vars in independency_graph.kset]
        for child in children:
            product_node.add_child(child)

        #for independent_set_of_vars in independency_graph.kset:
           # Only the training data with these variables
        #   product_node.add_child(detach_independent_vars(data, var_id_to_array_index,
        #   independent_set_of_vars, var_id_to_var, kclusters, pval, numeric))

        return product_node

    # Else we perform k-clustering on the instances
    logging.info("No independence found. Preparing for clustering...")

    _, unique_data_indices = numpy.unique(data, axis=0, return_index=True)

    if len(data) < kclusters or len(unique_data_indices) < kclusters:
        return fully_factorized_form(scope, data, var_id_to_array_index)

    clusters = kmedoid(kclusters, data, unique_data_indices)

    if kclusters == 1:
        return fully_factorized_form(scope, data, var_id_to_array_index)

    logging.info("Reformatting clusters to appropriate format and creating sum node...")
    sum_node = SumNode()
    with ProcessPoolExecutor(max_workers=2) as executor:
        children_weights = [
            (executor.submit(gens, scope, cluster, kclusters, pval, numeric),
            len(cluster)/len(data))
            for cluster in clusters
        ]
        children_weights = [(c.result(), w) for c, w in children_weights]
    for child, weight in children_weights:
        sum_node.add_child(child, weight)
    #for cluster in clusters:
    #    sum_node.add_child(
    #        gens(scope, cluster, kclusters, pval, numeric), len(cluster) / len(data)
    #    )
    return sum_node
