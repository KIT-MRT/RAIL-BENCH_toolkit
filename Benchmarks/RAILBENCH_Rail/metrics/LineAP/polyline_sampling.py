import numpy as np

from Benchmarks.RAILBENCH_Rail.metrics.LineAP.polyline_tools import polyline_to_oriented_points, polyline_to_oriented_segments


def point_sampling(rails, sample_distance, midpoints=True):
    """
    Transforms list of polylines to a list of sampled points.


    Parameters
    ----------
    rails : list
        List of polylines. Each polyline consists of a list of points.
        Example: [[[u1, v1], [u2, v2], ...], [[u1, v1], [u2, v2], ...], ...]

    sample_distance : int
        Distance between sampled points along the polylines.

    midpoints : bool
        If True, the midpoints of the segments are returned. 
        
    Returns
    -------
    all_points : list
        List of sampled points. [[[u1, v1], [u2, v2], ...]

    orientations : list
        List of orientations of the sampled points. [orient1, orient2, ...]

    rail_index_list : list
        List of indices indicating which polyline the sampled point belongs to.
    """
    first_pts = True

    rail_index_list = []

    for i, r in enumerate(rails):
        pts = polyline_to_oriented_points(np.array(r), sample_distance, midpoints=midpoints, include_last_segment=True)
        if pts is None:
            continue

        if first_pts:
            all_points = pts
            first_pts = False
        else:
            all_points = np.vstack((all_points, pts))

        rail_index_list.extend([i] * len(pts))


    if first_pts:
        return None, None, rail_index_list
    else:
        return all_points[:, :2].tolist(), all_points[:, 2].tolist(), rail_index_list
    

def segment_sampling(rails, sample_distance):
    """
    Transforms list of polylines to a list of sampled segments.

    Parameters
    ----------
    rails : list
        List of polylines. Each polyline consists of a list of points.
        Example: [[[u1, v1], [u2, v2], ...], [[u1, v1], [u2, v2], ...], ...]

    sample_distance : int
        Length of sampled segments. 

    Returns
    -------
    all_segments : list
        List of sampled segments.

    all_orientations : list
        List of orientations of the sampled segments.

    rail_index_list : list
        List of indices indicating which polyline the sampled segment belongs to.

    """

    first_segments = True

    rail_index_list = []

    for i, r in enumerate(rails):
        segments, orientations = polyline_to_oriented_segments(np.array(r), sample_distance)
        if segments is None:
            continue

        if first_segments:
            all_segments = segments
            all_orientations = orientations
            first_segments = False
        else:
            #all_segments = np.vstack((all_segments, segments))
            all_segments.extend(segments)
            all_orientations.extend(orientations)

        rail_index_list.extend([i] * len(segments))

    if first_segments:
        return None, None, rail_index_list
    else:
        return all_segments, all_orientations, rail_index_list

