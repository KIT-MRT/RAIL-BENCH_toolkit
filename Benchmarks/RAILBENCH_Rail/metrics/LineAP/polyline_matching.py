import numpy as np

from shapely import LineString, Point

import networkx as nx
from networkx.algorithms import bipartite
from scipy.spatial import distance
from collections import defaultdict




def point_segment_matching(pred_segments, gt_points, pred_orient, gt_orient, 
                           dist_thres, orient_thres, pred_confidence=None, matching_strategy='min_w_maximal_matching', use_confidence=True):
    """
    Takes a set of gt points and a set of predicted segments and computes a matching between them based on the euclidean distance and the orientation difference.

    Note: the consider_total_costs parameter heavily influences the runtime of the algorithm. (True: slower, False: faster)

    Parameters
    ----------
    pred_segments : list
        list of predicted segments [seg1, seg2, ..., segN]. Each segment is a list of points. 
        Example: seg1 = [[u1, v1], [u2, v2], [u3, v3]]; seg2 = [[u1, v1], [u2, v2]]

    gt_points : list
        list of sampled GT points. Each point is represented as [u, v]

    pred_orient : list
        list of orientations of the predicted segments.

    gt_orient : list
        list of orientations of the line segments the GT points are associated with.


    dist_thres : float
        Distance threshold for the matching. 
        A point can only be matched to a segment, if the shortest distance between the point and the segment is below this threshold.

    orient_thres : float
        Orientation threshold for the matching. 
        A point can only be matched to a segment, if the orientation difference between the segment the gt-point belongs to
        and the predicted segment is below this threshold.

    pred_confidence : list, optional
        List of confidence scores for the predicted segments. required if use_confidence is True.

    matching_strategy : str
        Matching strategy. 
        'maximum_matching': compute the maximum number of matches, and ignores the euclidean distance. (hopcroft_karp_matching)
        'min_w_maximum_matching': compute the maximum number of matches while minimizing the total euclidean distance of all matches. 
            (Among all maximum matchings, finds the one with the smallest possible total edge weight.)
        'min_w_maximal_matching': among all maximal matches, finds the one with the smallest possible total edge weight.
        
    use_confidence : bool
        If True, the confidence scores are used to sort the segments before matching.
        If False, the segments are matched without considering the confidence scores.

    Returns
    -------
    match_pred_ind : list
        List of indices of the matched predicted points.

    match_gt_ind : list
        List of indices of the matched GT points.

    avg_matching_distance : float
        Average matching distance of the matched points. 
        If no points are matched, returns -1.
        
    """
    assert matching_strategy in ['maximum_matching', 'min_w_maximum_matching', 'min_w_maximal_matching'], "Invalid matching strategy."
    assert not use_confidence or pred_confidence is not None, "pred_confidence must be provided for confidence_matching strategy."  

    if use_confidence:
        assert all(pred_confidence[i] >= pred_confidence[i+1] for i in range(len(pred_confidence)-1)), "pred_confidence must be sorted in descending order."

    # transform to shapely objects
    pred_segments_shapely = []
    for seg in pred_segments:
        line = LineString(seg)
        pred_segments_shapely.append(line)

    gt_points_shapely = []
    for pt in gt_points:
        point = Point(pt)
        gt_points_shapely.append(point)

    # compute shortest distances between GT points and predicted segments
    euclidean_distances = np.full((len(pred_segments_shapely), len(gt_points_shapely)), np.inf)
    for i, l in enumerate(pred_segments_shapely):
        for j, p in enumerate(gt_points_shapely):
            euclidean_distances[i, j] = l.distance(p)

    # compute angular distances of GT segments and predicted segments
    orient_differences = np.full((len(pred_orient), len(gt_orient)), np.inf)
    gt_orient = np.array(gt_orient)
    for idx_pred, o in enumerate(pred_orient):
        a = np.abs(gt_orient - o)
        b = np.abs(np.abs(gt_orient - o) - 360)
        orient_differences[idx_pred] = np.min(np.stack((a, b)), axis=0)

    if use_confidence:
        matches = confidence_matching(cost_matrix = euclidean_distances.copy(),
                                       orient_matrix = orient_differences,
                                       pred_confidence = pred_confidence,
                                       dist_thres = dist_thres,
                                       orient_thres = orient_thres,
                                       sub_matching_strategy=matching_strategy)
    else:
        matches = graph_matching(cost_matrix = euclidean_distances.copy(),
                                 orient_matrix = orient_differences,
                                 dist_thres = dist_thres,
                                 orient_thres = orient_thres,
                                 task=matching_strategy)
    
    if len(matches) == 0:
        return [], [], -1
    
    match_pred_ind, match_gt_ind = zip(*matches)
    match_pred_ind = list(match_pred_ind)
    match_gt_ind = list(match_gt_ind)

    total_costs = 0
    for i, j in matches:
        total_costs += euclidean_distances[i, j]
    avg_matching_distance = total_costs / len(matches)

    return match_pred_ind, match_gt_ind, avg_matching_distance


def pointwise_matching(pred_points, target_points, dist_thres, orient_thres, matching_strategy='min_w_maximum_matching'):
    """
    Takes two sets of points and computes the pointwise matching between them based on the euclidean distance and the orientation difference.

    Note: the consider_total_costs parameter heavily influences the runtime of the algorithm. (True: slower, False: faster)

    Parameters
    ----------
    pred_points : np.array
        shape: (n, 3) where n is the number of points and the columns are [u, v, orientation]
    target_points : np.array
        shape: (m, 3) where m is the number of points and the columns are [u, v, orientation]
    dist_thres : float
        Distance threshold for the matching. 
        Only points with a distance below this threshold can be matched with each other. 
    orient_thres : float
        Orientation threshold for the matching. 
        Only points with an orientation difference below this threshold can be matched with each other.
    matching_strategy : str
        Matching strategy. 
        'maximum_matching': compute the maximum number of matches, and ignores the euclidean distance. (hopcroft_karp_matching)
        'min_w_maximum_matching': compute the maximum number of matches while minimizing the total euclidean distance of all matches. 
            (Among all maximum matchings, finds the one with the smallest possible total edge weight.)
        'min_w_maximal_matching': among all maximal matches, finds the one with the smallest possible total edge weight.

    Returns
    -------
    match_pred_ind : list
        List of indices of the matched predicted points.
    match_gt_ind : list
        List of indices of the matched GT points.
    avg_matching_distance : float
        Average matching distance of the matched points. 
        If no points are matched, returns 0.
    """

    euclidean_distances = distance.cdist(pred_points[:, :2], target_points[:, :2], 'euclidean')
    orient_differences = np.full((len(pred_points), len(target_points)), np.inf)

    for idx_pred, pred_point in enumerate(pred_points):
        # compute angular distances of predicted point to all GT points
        a = np.abs(target_points[:, 2] - pred_point[2])
        b = np.abs(np.abs(target_points[:, 2] - pred_point[2]) - 360)
        orient_differences[idx_pred] = np.min(np.stack((a, b)), axis=0)

    matches = graph_matching(cost_matrix = euclidean_distances,
                                 orient_matrix = orient_differences,
                                 dist_thres = dist_thres,
                                 orient_thres = orient_thres,
                                 task=matching_strategy)
    
    if len(matches) == 0:
        return [], [], 0
    
    match_pred_ind, match_gt_ind = zip(*matches)
    match_pred_ind = list(match_pred_ind)
    match_gt_ind = list(match_gt_ind)

    total_costs = 0
    for i, j in matches:
        total_costs += euclidean_distances[i, j]
    avg_matching_distance = total_costs / len(matches)

    return match_pred_ind, match_gt_ind, avg_matching_distance



def graph_matching(cost_matrix, orient_matrix, dist_thres=10, orient_thres=10, task='min_w_maximal_matching'):
    """
    Builds a bipartite graph and computes matches. 

    Graph: 
    If the entries (i,j) in the cost_matrix and the orient_matrix are below the corresponding thresholds, an edge is added between the nodes i and j.

    Matching: 
    Different matching algorithms can be used.
    Note: 
    A maximum matching is a matching that contains the largest possible number of edges.
    A maximal matching is a matching that cannot be extended by adding another edge. 
    A maximal matching is not necessarily maximum — it might have fewer edges than the maximum possible matching.

    maximum matching -> compute the maximum number of matches, and ignores the euclidean distance. (hopcroft_karp_matching)
    minimum-weight maximum matching -> compute the maximum number of matches while minimizing the total euclidean distance of all matches. 
        (Among all maximum matchings, finds the one with the smallest possible total edge weight.)
    minimum-weight maximal matching -> among all maximal matches, finds the one with the smallest possible total edge weight.

    """

    assert task in ['maximum_matching', 'min_w_maximum_matching', 'min_w_maximal_matching'], "Invalid task."

    num_rows, num_cols = cost_matrix.shape
    G = nx.Graph()

    plotting = False

    if task == 'maximum_matching' or plotting:
        left_set = set()
        # right_set = set()

    # row_appeard = defaultdict(int)
    # col_appeard = defaultdict(int)

    # Add edges only for valid pairs (cost below threshold)
    for i in range(num_rows):
        for j in range(num_cols):
            if cost_matrix[i, j] <= dist_thres and orient_matrix[i, j] <= orient_thres:
                G.add_edge(f"row_{i}", f"col_{j}", weight=cost_matrix[i, j])  

                if task == 'maximum_matching' or plotting:
                    left_set.add(f"row_{i}")
                    # right_set.add(f"col_{j}")

                # row_appeard[i] += 1
                # col_appeard[j] += 1

    # for i, count in row_appeard.items():
    #     if count > 1:
    #         print(f"Warning: row_{i} appears in {count} edges.")
    # for j, count in col_appeard.items():
    #     if count > 1:
    #         print(f"Warning: col_{j} appears in {count} edges.")

    if plotting:  # visualize graph
        import matplotlib.pyplot as plt

        nx.draw(G, with_labels=True, pos=nx.bipartite_layout(G, left_set))

        pos = nx.spring_layout(G, seed=7)
        nx.draw_networkx_nodes(G, pos, node_size=300)
        edges = [(u, v) for (u, v, d) in G.edges(data=True)]
        nx.draw_networkx_edges(G, pos, edgelist=edges, width=6)
        nx.draw_networkx_labels(G, pos, font_size=20, font_family="sans-serif")
        #edge_labels = nx.get_edge_attributes(G, "weight")
        #nx.draw_networkx_edge_labels(G, pos, edge_labels)

        ax = plt.gca()
        ax.margins(0.08)
        plt.axis("off")
        plt.tight_layout()
        plt.show()


    if task == 'maximum_matching':
        matching = bipartite.hopcroft_karp_matching(G, left_set)
        matches = [(int(r[4:]), int(c[4:])) if r.startswith("row") else (int(c[4:]), int(r[4:])) 
            for r, c in matching.items() if r.startswith("row")]
        
    elif task == 'min_w_maximum_matching':
        matching = nx.min_weight_matching(G)
        matches = [(int(r[4:]), int(c[4:])) if r.startswith("row") else (int(c[4:]), int(r[4:])) for r, c in matching]

    elif task == 'min_w_maximal_matching':
        matching = minimal_weight_maximal_matching(G)
        matches = [(int(r[4:]), int(c[4:])) if r.startswith("row") else (int(c[4:]), int(r[4:])) for r, c in matching]

    return matches

def minimal_weight_maximal_matching(G):
    """
    Computes the minimum-weight maximal matching of a graph.
    """
    
    # Sort edges by weight (ascending order)
    sorted_edges = sorted(G.edges(data=True), key=lambda x: x[2]['weight'])

    matching = set()
    matched_nodes = set()

    # Greedily select edges ensuring maximality with minimal weight
    for u, v, _ in sorted_edges:
        if u not in matched_nodes and v not in matched_nodes:
            matching.add((u, v))
            matched_nodes.add(u)
            matched_nodes.add(v)

    return matching


def confidence_matching(cost_matrix, orient_matrix, pred_confidence, dist_thres=10, orient_thres=10, sub_matching_strategy='min_w_maximum_matching'):
    """
    Computes the matches iteratively starting with the predictions with the highest confidence score.
    For each predictions with the same confidence score, the matches are computed using the sub_matching_strategy.   
    """

    all_matches = []

    # iterate over all confidence scores and compute the matches
    unique_confidences = np.sort(np.unique(pred_confidence))[::-1]

    for c in unique_confidences:
        # get the indices of the segments with the current confidence score
        indices = np.where(pred_confidence == c)[0]

        # filter the cost matrix and orientation matrix for the current confidence score
        cost_matrix_filtered = cost_matrix[indices, :]
        orient_matrix_filtered = orient_matrix[indices, :]

        # compute the matches for the current confidence score
        matches = graph_matching(cost_matrix=cost_matrix_filtered,
                                 orient_matrix=orient_matrix_filtered,
                                 dist_thres=dist_thres,
                                 orient_thres=orient_thres,
                                 task=sub_matching_strategy)
        # Note: the indices in matches for the predictions correspond to the filtered cost matrix !

        if len(matches) > 0:
            match_pred_ind_filtered, match_gt_ind = zip(*matches)
            # map indices back to original indices
            match_pred_ind = list(indices[np.array(match_pred_ind_filtered)])

            all_matches.extend(zip(match_pred_ind, match_gt_ind))

            # set the columns of the matched ground truth points from the cost matrix and orientation matrix to inf (to avoid matching them again)
            cost_matrix[:, match_gt_ind] = np.inf
            orient_matrix[:, match_gt_ind] = np.inf


    return all_matches