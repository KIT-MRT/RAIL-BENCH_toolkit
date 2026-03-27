"""
This file was developed with the assistance of AI-based tools (e.g., Claude). 
All content has been reviewed and adapted by the author, but AI-generated 
contributions may be present.
"""

import numpy as np
from scipy.spatial import cKDTree

def sample_polyline(lane, num_points=100):
    """Uniformly sample points along a polyline by arc length."""
    pts = np.array(lane, dtype=float)
    diffs = np.diff(pts, axis=0)
    seg_lengths = np.linalg.norm(diffs, axis=1)
    cumlen = np.concatenate([[0], np.cumsum(seg_lengths)])
    total = cumlen[-1]
    if total == 0:
        return pts[[0]].repeat(num_points, axis=0)
    sample_dists = np.linspace(0, total, num_points)
    sampled = []
    for d in sample_dists:
        idx = np.searchsorted(cumlen, d, side='right') - 1
        idx = np.clip(idx, 0, len(pts) - 2)
        t = (d - cumlen[idx]) / (seg_lengths[idx] + 1e-9)
        sampled.append(pts[idx] + t * diffs[idx])
    return np.array(sampled)

def chamfer_distance_polylines(pred_lane, gt_lane, num_points=100):
    """Chamfer distance between two polylines."""
    pred_pts = sample_polyline(pred_lane, num_points)
    gt_pts   = sample_polyline(gt_lane,   num_points)

    tree_gt   = cKDTree(gt_pts)
    tree_pred = cKDTree(pred_pts)

    d_pred_to_gt, _ = tree_gt.query(pred_pts)
    d_gt_to_pred, _ = tree_pred.query(gt_pts)

    return 0.5 * (d_pred_to_gt.mean() + d_gt_to_pred.mean())

def chamfer_metric(pred_lanes, gt_lanes, num_points=100):
    """
    Compute mean Chamfer Distance across all (pred, gt) lane pairs.
    Uses Hungarian matching to find optimal assignment.
    """
    from scipy.optimize import linear_sum_assignment

    n_pred, n_gt = len(pred_lanes), len(gt_lanes)
    if n_pred == 0 and n_gt == 0:
        return {"mean_chamfer": 0.0}
    if n_pred == 0 or n_gt == 0:
        return {"mean_chamfer": float('inf')}

    cost = np.zeros((n_pred, n_gt))
    for i, pl in enumerate(pred_lanes):
        for j, gl in enumerate(gt_lanes):
            cost[i, j] = chamfer_distance_polylines(pl, gl, num_points)

    row_ind, col_ind = linear_sum_assignment(cost)
    matched_costs = [cost[r, c] for r, c in zip(row_ind, col_ind)]

    # Penalize unmatched lanes with max observed distance
    penalty = max(matched_costs) if matched_costs else 0.0
    n_unmatched = abs(n_pred - n_gt)
    all_costs = matched_costs + [penalty] * n_unmatched

    return {"mean_chamfer": np.mean(all_costs),
            "matched_chamfer": np.mean(matched_costs) if matched_costs else 0.0,
            "per_pair": list(zip(row_ind.tolist(), col_ind.tolist(), matched_costs))}