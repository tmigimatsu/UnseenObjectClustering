# Copyright (c) 2020 NVIDIA Corporation. All rights reserved.
# This work is licensed under the NVIDIA Source Code License - Non-commercial. Full
# text can be found in LICENSE.md

import torch
import torch.nn.functional as F
import numpy as np
from fcn.config import cfg

def ball_kernel(Z, X, kappa, metric='cosine'):
    """ Computes pairwise ball kernel (without normalizing constant)
        (note this is kernel as defined in non-parametric statistics, not a kernel as in RKHS)

        @param Z: a [n x d] torch.FloatTensor of NORMALIZED datapoints - the seeds
        @param X: a [m x d] torch.FloatTensor of NORMALIZED datapoints - the points

        @return: a [n x m] torch.FloatTensor of pairwise ball kernel computations,
                 without normalizing constant
    """
    if metric == 'euclidean':
        distance = Z.unsqueeze(1) - X.unsqueeze(0)
        distance = torch.norm(distance, dim=2)
        kernel = torch.exp(-kappa * torch.pow(distance, 2))
    elif metric == 'cosine':
        kernel = torch.exp(kappa * torch.mm(Z, X.t()))
    return kernel


def get_label_mode(array):
    """ Computes the mode of elements in an array.
        Ties don't matter. Ties are broken by the smallest value (np.argmax defaults)

        @param array: a numpy array
    """
    labels, counts = np.unique(array, return_counts=True)
    mode = labels[np.argmax(counts)].item()
    return mode


def connected_components(Z, epsilon, metric='cosine'):
    """
        For the connected components, we simply perform a nearest neighbor search in order:
            for each point, find the points that are up to epsilon away (in cosine distance)
            these points are labeled in the same cluster.

        @param Z: a [n x d] torch.FloatTensor of NORMALIZED datapoints

        @return: a [n] torch.LongTensor of cluster labels
    """
    n = Z.shape[0]

    K = 0
    cluster_labels = torch.full((n,), -1, dtype=torch.long, device=Z.device)

    if metric == "euclidian":
        distances = torch.norm(Z.unsqueeze(1) - Z.unsqueeze(0), dim=2)
    elif metric == "cosine":
        distances = 0.5 * (1 - torch.mm(Z, Z.t()))
    else:
        raise RuntimeError(f"Unsupported metric {metric}.")
    component_seeds = distances <= epsilon

    for i in range(n):
        if cluster_labels[i] != -1:
            continue

        # If at least one component already has a label, then use the mode of the label
        idx_cluster = component_seeds[i]
        labels, counts = torch.unique(cluster_labels[idx_cluster], return_counts=True)
        if labels.shape[0] > 1:
            counts[labels==-1] = 0
            label = labels[torch.argmax(counts)]
        else:
            label = K
            K += 1  # Increment number of clusters

        cluster_labels[idx_cluster] = label

    return cluster_labels


def seed_hill_climbing_ball(X, Z, kappa, max_iters=10, metric='cosine'):
    """ Runs mean shift hill climbing algorithm on the seeds.
        The seeds climb the distribution given by the KDE of X

        @param X: a [n x d] torch.FloatTensor of d-dim unit vectors
        @param Z: a [m x d] torch.FloatTensor of seeds to run mean shift from
        @param dist_threshold: parameter for the ball kernel
    """
    for _ in range(max_iters):
        W = ball_kernel(Z, X, kappa, metric=metric)

        # use this allocated weight to compute the new center
        new_Z = torch.mm(W, X)  # Shape: [n x d]

        # Normalize the update
        if metric == 'euclidean':
            summed_weights = W.sum(dim=1)
            summed_weights = summed_weights.unsqueeze(1)
            summed_weights = torch.clamp(summed_weights, min=1.0)
            Z = new_Z / summed_weights
        elif metric == 'cosine':
            Z = F.normalize(new_Z, p=2, dim=1)

    return Z


def mean_shift_with_seeds(X, Z, kappa, max_iters=10, metric='cosine'):
    """ Runs mean-shift

        @param X: a [n x d] torch.FloatTensor of d-dim unit vectors
        @param Z: a [m x d] torch.FloatTensor of seeds to run mean shift from
        @param dist_threshold: parameter for the von Mises-Fisher distribution
    """
    Z = seed_hill_climbing_ball(X, Z, kappa, max_iters=max_iters, metric=metric)

    # Connected components
    cluster_labels = connected_components(Z, 2 * cfg.TRAIN.EMBEDDING_ALPHA, metric=metric)  # Set epsilon = 0.1 = 2*alpha

    return cluster_labels, Z


def select_smart_seeds(X, num_seeds, return_selected_indices=False, init_seeds=None, num_init_seeds=None, metric='cosine'):
    """ Selects seeds that are as far away as possible

        @param X: a [n x d] torch.FloatTensor of d-dim unit vectors
        @param num_seeds: number of seeds to pick
        @param init_seeds: a [num_seeds x d] vector of initial seeds
        @param num_init_seeds: the number of seeds already chosen.
                               the first num_init_seeds rows of init_seeds have been chosen already

        @return: a [num_seeds x d] matrix of seeds
                 a [n x num_seeds] matrix of distances
    """
    n, d = X.shape
    selected_indices = torch.full((num_seeds,), -1, dtype=torch.long, device=X.device)

    # Initialize seeds matrix
    if init_seeds is None:
        seeds = torch.empty((num_seeds, d), device=X.device)
        num_chosen_seeds = 0
    else:
        seeds = init_seeds
        num_chosen_seeds = num_init_seeds

    # Keep track of distances
    distances = torch.empty((num_seeds, n), device=X.device)

    if num_chosen_seeds == 0:  # Select first seed if need to
        selected_seed_index = np.random.randint(0, n)
        selected_indices[0] = selected_seed_index
        selected_seed = X[selected_seed_index, :]
        seeds[0] = selected_seed
        if metric == 'euclidean':
            distances[0] = torch.norm(X - selected_seed.unsqueeze(0), dim=1)
        elif metric == 'cosine':
            distances[0] = 0.5 * (1 - torch.matmul(X, selected_seed))
        num_chosen_seeds += 1
    else:  # Calculate distance to each already chosen seed
        for i in range(num_chosen_seeds):
            if metric == 'euclidean':
                distances[i] = torch.norm(X - seeds[i:i+1, :], dim=1)
            elif metric == 'cosine':
                distances[i] = 0.5 * (1 - torch.mm(X, seeds[i:i+1, :].t())[:, 0])

    # Select rest of seeds
    distance_to_nearest_seed = torch.min(distances[:num_chosen_seeds], dim=0)[0]  # Shape: [n]
    for i in range(num_chosen_seeds, num_seeds):
        # Find the point that has the furthest distance from the nearest seed
        # distance_to_nearest_seed = torch.min(distances[:i], dim=0)[0]  # Shape: [n]
        selected_seed_index = torch.argmax(distance_to_nearest_seed)
        selected_indices[i] = selected_seed_index
        selected_seed = torch.index_select(X, 0, selected_seed_index).squeeze()
        seeds[i] = selected_seed

        # Calculate distance to this selected seed
        if metric == 'euclidean':
            distances[i] = torch.norm(X - selected_seed.unsqueeze(0), dim=1)
        elif metric == 'cosine':
            distances[i] = 0.5 * (1 - torch.matmul(X, selected_seed))
        distance_to_nearest_seed = torch.minimum(distance_to_nearest_seed, distances[i])

    return_tuple = (seeds,)
    if return_selected_indices:
        return_tuple += (selected_indices,)
    return return_tuple


def mean_shift_smart_init(X, kappa, num_seeds=100, max_iters=10, metric='cosine'):
    """ Runs mean shift with carefully selected seeds

        @param X: a [n x d] torch.FloatTensor of d-dim unit vectors
        @param dist_threshold: parameter for the von Mises-Fisher distribution
        @param num_seeds: number of seeds used for mean shift clustering

        @return: a [n] array of cluster labels
    """
    seeds, selected_indices = select_smart_seeds(X, num_seeds, return_selected_indices=True, metric=metric)
    seed_cluster_labels, updated_seeds = mean_shift_with_seeds(X, seeds, kappa, max_iters=max_iters, metric=metric)

    # Get distances to updated seeds
    if metric == 'euclidean':
        distances = X.unsqueeze(1) - updated_seeds.unsqueeze(0)  # a are points, b are seeds
        distances = torch.norm(distances, dim=2)
    elif metric == 'cosine':
        distances = 0.5 * (1 - torch.mm(X, updated_seeds.t())) # Shape: [n x num_seeds]

    # Get clusters by assigning point to closest seed
    closest_seed_indices = torch.argmin(distances, dim=1)  # Shape: [n]
    cluster_labels = seed_cluster_labels[closest_seed_indices]

    # assign zero to the largest cluster
    unique_labels, count = torch.unique(seed_cluster_labels, return_counts=True)
    label_max = torch.index_select(unique_labels, 0, torch.argmax(count))
    if label_max != 0:
        index1 = cluster_labels == 0
        index2 = cluster_labels == label_max
        cluster_labels[index1] = label_max
        cluster_labels[index2] = 0

    return cluster_labels, selected_indices
