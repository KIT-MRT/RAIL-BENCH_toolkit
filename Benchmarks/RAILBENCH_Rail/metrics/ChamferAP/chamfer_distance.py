"""
This file was developed with the assistance of AI-based tools (e.g., Claude). 
All content has been reviewed and adapted by the author, but AI-generated 
contributions may be present.
"""

import numpy as np
from scipy.spatial.distance import cdist

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
    idx = np.searchsorted(cumlen, sample_dists, side='right') - 1
    idx = np.clip(idx, 0, len(pts) - 2)
    t = (sample_dists - cumlen[idx]) / (seg_lengths[idx] + 1e-9)
    return pts[idx] + t[:, None] * diffs[idx]

def chamfer_distance_polylines(pred_lane, gt_lane, num_points=100):
    """Chamfer distance between two polylines."""
    pred_pts = sample_polyline(pred_lane, num_points)
    gt_pts   = sample_polyline(gt_lane,   num_points)

    dists = cdist(pred_pts, gt_pts)
    d_pred_to_gt = dists.min(axis=1)
    d_gt_to_pred = dists.min(axis=0)

    return 0.5 * (d_pred_to_gt.mean() + d_gt_to_pred.mean())
