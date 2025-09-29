
from scipy.spatial.distance import pdist, squareform


def calculate_distance_matrix(features, metric='euclidean'):
    """Calculate the pairwise distance matrix for a set of features."""
    return squareform(pdist(features, metric))


def compute_top_percentile_mask(distance_matrix, percentile=50):
    """Create a binary mask where only the top percentile distances are marked."""
    threshold = np.percentile(distance_matrix, percentile)
    return (distance_matrix >= threshold).astype(int)



def intersect_masks(masks):
    """Intersect multiple binary masks."""
    intersection_mask = masks[0]
    for mask in masks[1:]:
        intersection_mask = np.logical_and(intersection_mask, mask)
    return intersection_mask



def process_views(feature_list, percentile=90):
    """Process all views to compute the final cannot-link matrix based on the intersection of top percentile distances."""
    masks = []
    for features in feature_list:
        dist_matrix = calculate_distance_matrix(features)
        mask = compute_top_percentile_mask(dist_matrix, percentile)
        masks.append(mask)

    cannot_link_matrix = intersect_masks(masks)
    return cannot_link_matrix

import numpy as np


def compute_top_percentile_score(dist_matrix, percentile):
    """
    Compute a score matrix based on the top percentile distances in the distance matrix.
    Points with distances in the top 'percentile' will receive higher scores.

    :param dist_matrix: Distance matrix (numpy array)
    :param percentile: Percentile threshold for scoring distances (e.g., 90 means top 10% of distances get higher scores)
    :return: A score matrix where higher scores indicate greater distance (cannot-link)
    """
    threshold = np.percentile(dist_matrix, percentile)
    score_matrix = np.zeros_like(dist_matrix)
    max_dist = np.max(dist_matrix)
    min_dist = threshold
    score_matrix[dist_matrix > threshold] = (dist_matrix[dist_matrix > threshold] - min_dist) / (max_dist - min_dist)

    return score_matrix


