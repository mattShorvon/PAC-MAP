"""Kmedoid: Contains the PAM algorithm for clustering on Kmedoids."""

# Code based on the article: Bauckhage, C. (2015). Numpy/scipy Recipes for Data
# Science: k-Medoids Clustering. researchgate. net, Feb.

from typing import List
import numpy
from scipy.spatial import distance


def kmedoid(
        kclusters: int, data: numpy.array, unique_data_indices: numpy.array, max_iterations: int = 100
) -> List[numpy.array]:
    """PAM algorithm for kmedoid clustering"""
    distance_matrix = distance.squareform(distance.pdist(data, "sqeuclidean"))

    medoid_indexes = numpy.random.choice(unique_data_indices, kclusters, replace=False)
    medoid_indexes.sort()
    new_medoid_indexes = numpy.copy(medoid_indexes)

    clusters = {}

    indexes_of_clusters = numpy.argmin(distance_matrix[:, medoid_indexes], axis=1)

    for _ in range(max_iterations):
        for medoid_index in range(kclusters):
            clusters[medoid_index] = numpy.where(indexes_of_clusters == medoid_index)[0]

        for medoid_index in range(kclusters):
            indexes_on_distance_matrix = numpy.ix_(
                clusters[medoid_index], clusters[medoid_index]
            )
            mean_distance = numpy.mean(
                distance_matrix[indexes_on_distance_matrix], axis=1
            )
            minimum_distance_index = numpy.argmin(mean_distance)

            new_medoid_indexes[medoid_index] = clusters[medoid_index][
                minimum_distance_index
            ]
        numpy.sort(new_medoid_indexes)

        if numpy.array_equal(new_medoid_indexes, medoid_indexes):
            # Convergence
            break

        medoid_indexes = numpy.copy(new_medoid_indexes)

    final_result = [data[data_indexes] for data_indexes in clusters.values()]
    return final_result
