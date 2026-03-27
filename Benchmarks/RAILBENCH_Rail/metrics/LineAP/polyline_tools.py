import numpy as np
from matplotlib import pyplot as plt

from shapely.geometry import LineString

from collections import defaultdict



def orientation_polyline(polyline):
    """
    Compute the orientation for each line segment in the provided polyline.
    Here a line segment is defined by two consecutive points in the polyline.

    Parameters
    ----------
    line : np.array (N, 2)
        The polyline to compute the orientation for.
    
    Returns
    -------
    direction_deg : np.array (N-1,)
        The orientation of each line segment in degrees.
        The orientation starts at 0 degrees and increases clockwise. 
        0 degrees is the positive x-axis/u-axis.
    """
    return orientation_start_end_point(polyline[:-1], polyline[1:])

def orientation_start_end_point(startpoints, endpoints):
    """
    Compute the orientation of lines defined by start and endpoints

    Parameters
    ----------
    startpoints: np.array(K, 2)
        startpoints of K line segments

    endpoints: np.array(K, 2)
        endpoints of K line segments

    Returns
    -------
    direction_deg : np.array (K,)
        The orientation of each line segment in degrees.
        The orientation starts at 0 degrees and increases clockwise. 
        0 degrees is the positive x-axis/u-axis.
    """
    vectors = endpoints - startpoints
    direction_rad = np.arctan2(vectors[:, 1], vectors[:, 0])
    direction_rad[direction_rad < 0] = 2 * np.pi + direction_rad[direction_rad < 0]
    direction_deg =  np.degrees(direction_rad)

    return direction_deg
    
#-------------------------------------------------------------------------------
def polyline_to_points(polyline, sample_distance, include_last_pt=True):
    """
    Convert a polyline to equally sampled points along the line.
    
    Input:
    polyline : np.array (N, 2)
        The polyline to sample points from.

    sample_distance : float
        The distance between each sampled point.

    include_last_pt : bool
        If True, the last point will be included even if the last segment is shorter than sample_distance.
    """
    ls = LineString(polyline)
    if ls.length < sample_distance:
        if include_last_pt:
            return np.vstack([polyline[0], polyline[-1]])
        else:
            return np.array(polyline[0]).reshape(1, 2)

    distances = np.arange(0, ls.length, sample_distance)
    if ls.length - distances[-1] == sample_distance:
        distances = np.append(distances, ls.length)
    points = np.array([ls.interpolate(dist).coords[0] for dist in distances])

    if include_last_pt and (ls.length != distances[-1]):
        points = np.vstack([points, ls.interpolate(ls.length).coords[0]])

    return points

def polyline_to_midpoints(polyline, sample_distance, include_last_segment=True):
    """
    Divide polyline into equally long segments (except for last segment, which is commonly shorter) and return the midpoint of each segment.
    
    Parameters:
    -----------
    polyline : np.array (N, 2)
        The polyline to sample points from.

    sample_distance : float
        The distance between each sampled point.

    include_last_segment : bool
        specify whether to include a midpoint for the last segment or not if the last segment is shorter than sample_distance.

    Returns:
    --------
    points : np.array (N, 2)
        The midpoints of the polyline.
    """
    ls = LineString(polyline)
    
    if ls.length < sample_distance:
        if include_last_segment:
            return np.array([ls.interpolate(ls.length/2).coords[0]])
        else:
            return None

    distances = np.arange(sample_distance/2, ls.length, sample_distance)

    l_remaining = ls.length % sample_distance
    if l_remaining != 0:
        if l_remaining > sample_distance/2:
            distances = distances[:-1]

        if include_last_segment:
            d_to_add = l_remaining/2 + sample_distance/2 + distances[-1]
            distances = np.hstack([distances, d_to_add])

    points = np.array([ls.interpolate(dist).coords[0] for dist in distances])

    return points

def polyline_to_oriented_points(polyline, sample_distance, midpoints=False, include_last_segment=True, include_last_pt=True):
    """
    Divides a polyline into segments of size sample_distance and returns the start/end points or midpoints of each segment. 
    Additional it computes an orientation for each point based on the orientation of the line segment the point is associated with.

    If midpoints is True, the points are sampled at the midpoints of the line segments. 
    If midpoints is False, the points are sampled at the start of the line segments.

    If include_last_segment is True, the last segment is included even if it is shorter than sample_distance (only relevant if midpoints is True).
    If include_last_pt is True, the last point of the polyline is included even if the last segment is shorter than sample_distance (only relevant if midpoints is False).

    Parameters:
    -----------
    polyline : np.array (N, 2)
        The polyline to sample points from.

    sample_distance : float
        The distance between each sampled point.

    midpoints : bool
        If True, the points are sampled at the midpoints of the line segments.
        If False, the points are sampled at the start of the line segments.

    include_last_segment : bool
        only relevant if midpoints is True.
        specify whether to include the last segment or not if the last segment is shorter than sample_distance.

    include_last_pt : bool
        only relevant if midpoints is False.
        If True, include the last point of the polyline even if the last segment is shorter than sample_distance.

    Returns:
    --------
    points : np.array (N, 3)
        The points of the polyline with an orientation for each point.
    """
    if midpoints:
        midpoints = polyline_to_midpoints(polyline, sample_distance, include_last_segment=include_last_segment)

        if midpoints is None:
            return None

        elif len(midpoints) == 1:
            direction = orientation_polyline(np.vstack([polyline[0], polyline[-1]]))
            return np.hstack([midpoints, direction.reshape(-1, 1)])

        else:
            start_end_points = polyline_to_points(polyline, sample_distance, include_last_pt=include_last_segment)
            directions = orientation_polyline(start_end_points)
            return np.hstack([midpoints, directions.reshape(-1, 1)])
    else:
        points = polyline_to_points(polyline, sample_distance, include_last_pt=include_last_pt)

        if len(points) == 1:
            direction = orientation_polyline(np.vstack([polyline[0], polyline[-1]]))
            return np.hstack([points, direction.reshape(-1, 1)])
        
        else:
            directions = orientation_polyline(points)
            directions = np.hstack([directions, directions[-1]])
            return np.hstack([points, directions.reshape(-1, 1)])

#-------------------------------------------------------------------------------
def polyline_to_linesegments(polyline, sample_distance):
    """
    Sample line segments from polyline

    Parameters
    ----------
    polyline : np.array (N, 2)
        The polyline to sample points from.

    sample_distance : float
        The distance between each sampled point.

    Returns:
    ---------
    list of line segments where each line segment is represented by a list of points [[u1, v1], [u2, v2], ...]
    """
    # step 1: compute start and endpoints of all line segments
    points = polyline_to_points(polyline, sample_distance, include_last_pt=True)
    startpoints = points[:-1]
    endpoints = points[1:] 

    # Note: the sampled line segments should also include the anchor points of the original polyline

    # step 2: Assign anchor points of polyline to line segments 
    s = 0
    i_seg = 0 # index of newly sampled segments
    pt_add = defaultdict(list)

    # iterate over original line segments in the polyline
    for i_orig in range(len(polyline)-2):
        length_orig_seg = np.linalg.norm(polyline[i_orig+1] - polyline[i_orig])
        pt = list(polyline[i_orig+1]) # endpoint of original line segment

        length = length_orig_seg  - s # s: remaining length of the line segment that started in the previous original line segment
        if s != 0:
            if length > 0: # the segment that started in the previous original line segment ends 
                i_seg += 1  
            else: # the segment that started in the previous original line segment doesn't end 
                # special case: no point was sampled from the given original line_segment
                s -= length_orig_seg
                pt_add[int(i_seg)].append(pt)
                continue # skip to next original line segment

        # here: i_seg represents first segment that starts in the current original line segment
        #if length == sample_distance: # case 1: end of original line segment is identical to end of new line segment -> no point must be added 
        if length % sample_distance == 0: # case 1: end of original line segment is identical to end of a sampled line segment -> no point must be added 
            i_seg += length//sample_distance
            s = 0

        elif length < sample_distance: # case 2: sampled segment goes over original-polyline-point -> add this point to the newly sampled segment
            pt_add[int(i_seg)].append(pt)
            s = sample_distance - length # length that the current line segment (i_orig) reaches into the next original line segment

        else: # case 3: at least one new line segment is completely included in the given original line segment 
            n_seg_fit = int(length//sample_distance)
            i_seg += n_seg_fit

            pt_add[int(i_seg)].append(pt)

            remaining_length = length%sample_distance 
            s = sample_distance - remaining_length

    # step 3: combine start and endpoints with additional points
    new_lines = []
    for i, (start, end) in enumerate(zip(startpoints.tolist(), endpoints.tolist())):
        new_line = [start] + np.array(pt_add[i]).tolist() + [end]
        new_lines.append(new_line)

    return new_lines


def polyline_to_oriented_segments(polyline, sample_distance):
    """
    Sample line segments from polyline and compute the orientation. 

    Parameters
    ----------
    polyline : np.array (N, 2)
        The polyline to sample points from.

    sample_distance : float
        The distance between each sampled point.

    Returns:
    ---------
    list of line segments where each line segment is represented by a list of points [[u1, v1], [u2, v2], ...]

    list of orientations for each line segment
    """
    segments = polyline_to_linesegments(polyline, sample_distance)
    orientation = []

    for seg in segments:
        startpoint = np.array(seg[0]).reshape(1, 2)
        endpoint = np.array(seg[-1]).reshape(1, 2)
        orientation.append(orientation_start_end_point(startpoint, endpoint)[0])

    return segments, orientation


#-------------------------------------------------------------------------------
def vis_polyline_sampling(polyline, points):
    """
    Create visualization of the polyline and the sampled points.
    """
    # rightRail = is_rightRail(polyline)
    
    plt.plot(polyline[:, 0], polyline[:, 1], 'o-')
    for i, p in enumerate(points):
        plt.plot(p[0], p[1], 'ro')
        plt.text(p[0], p[1], f"{p[2]:.2f}°")
        # if i == 0:
        #     if rightRail:
        #         plt.annotate("Start", (p[0], p[1]))
        #     else:
        #         plt.annotate("Start", (p[0], p[1]))
        
        
    plt.xlabel('u')
    plt.ylabel('v')
    plt.gca().invert_yaxis()
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

def viz_line_segments(line_segments):
    """
    Visualize sampled line segments
    """
    for line in line_segments:
        l = np.array(line)
        plt.plot(l[:, 0], l[:, 1], 'o-')
        
    plt.xlabel('u')
    plt.ylabel('v')
    plt.gca().invert_yaxis()
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()
