import numpy as np


def get_track_dict(rails, track_ids, right_rail):
    """
    Get a dictionary of tracks with their left and right rails.

    Args:
        rails (list): List of polylines.
        track_ids (list): List of track IDs corresponding to each rail.
        right_rail (list): List indicating whether each rail is a right rail (1) or left rail (0).

    """
    track_ids_unique = sorted(list(set(track_ids)))
    tracks_dict = dict()
    for tr_id in track_ids_unique:
        tracks_dict[tr_id] = {'left': [], 'right': []}
        for id, is_right, r in zip(track_ids, right_rail, rails):
            if id == tr_id:
                if is_right:
                    tracks_dict[tr_id]['right'].append(r.copy())
                else:
                    tracks_dict[tr_id]['left'].append(r.copy())
    return tracks_dict


def get_track_width(left_rail_list, right_rail_list, v):
    """
    Get the track width at a specific v-coordinate for a given list of left and right rails.
    Note, a track commonly consists of a single left and right rail, but in some rare cases, a rail might be interrupted and have multiple segments. This function will handle such cases by checking all segments of the left and right rails.

    Args:
        left_rail_list (list): List of left rail polylines (each polyline is a list of (x, y) points)
        right_rail_list (list): List of right rail polylines (each polyline is a list of (x, y) points)
        v (float): The v-coordinate at which to calculate the track width

    Returns:
        The track width at the specified v-coordinate, or None if the v-coordinate is outside the range of the rails.
    """
    if len(left_rail_list) == 0 or len(right_rail_list) == 0:
        return None  # No rails to calculate width from
    else:
        for left_rail in left_rail_list:
            for right_rail in right_rail_list:
                # Check if the v-coordinate is within the range of both rails
                left_v_coords = [point[1] for point in left_rail]
                right_v_coords = [point[1] for point in right_rail]
                if min(left_v_coords) <= v <= max(left_v_coords) and min(right_v_coords) <= v <= max(right_v_coords):
                    # Interpolate the x-coordinates of the left and right rails at the given v-coordinate
                    anchor_left_before = [point for point in left_rail if point[1] <= v][0]
                    anchor_left_after = [point for point in left_rail if point[1] >= v][-1]
                    left_x = np.interp(v, [anchor_left_before[1], anchor_left_after[1]], [anchor_left_before[0], anchor_left_after[0]])

                    anchor_right_before = [point for point in right_rail if point[1] <= v][0]
                    anchor_right_after = [point for point in right_rail if point[1] >= v][-1]
                    right_x = np.interp(v, [anchor_right_before[1], anchor_right_after[1]], [anchor_right_before[0], anchor_right_after[0]])

                    track_width = abs(left_x - right_x)
                    return float(track_width)
                
    return None  # If no valid width was found for the given v-coordinate

def get_all_track_widths(tracks_dict, v):
    """
    Get the track widths at a specific v-coordinate for all tracks in the tracks_dict.
    """
    track_widths = {}
    for track_id, rail_dict in tracks_dict.items():
        w = get_track_width(rail_dict['left'], rail_dict['right'], v)
        if w is not None:
            track_widths[track_id] = w

    return track_widths


def get_track_v_range(tracks_dict):
    """
    Get the overall lowest and highest v-coordinates across all tracks in the tracks_dict that have both left and right rails.

    Args:
        tracks_dict (dict): Dictionary of tracks with their left and right rails.
    
    Returns:
        A tuple (highest_img_v, lowest_img_v) representing the overall highest and lowest v-coordinates across all tracks with both left and right rails.   
    """
    lowest_img_v = float('inf')
    highest_img_v = float('-inf')

    for rail_dict in tracks_dict.values():
        if len(rail_dict['left']) > 0 and len(rail_dict['right']) > 0:
            low_left_v, high_left_v = float('inf'), float('-inf')
            low_right_v, high_right_v = float('inf'), float('-inf')
            for left_rail in rail_dict['left']:
                v_min = left_rail[-1][1]
                v_max = left_rail[0][1]
                low_left_v = min(low_left_v, v_min)
                high_left_v = max(high_left_v, v_max)

            for right_rail in rail_dict['right']:
                v_min = right_rail[-1][1]
                v_max = right_rail[0][1]
                low_right_v = min(low_right_v, v_min)
                high_right_v = max(high_right_v, v_max)

            # get lowest point with both left and right rails! 
            lowest_v = max(low_left_v, low_right_v)
            highest_v = min(high_left_v, high_right_v)

            # in rare cases, the left and right rails do not overlap in v-coordinates, so we need to check for that
            if get_track_width(rail_dict['left'], rail_dict['right'], lowest_v):
                lowest_img_v = min(lowest_img_v, lowest_v)
            if get_track_width(rail_dict['left'], rail_dict['right'], highest_v):
                highest_img_v = max(highest_img_v, highest_v)

    return highest_img_v, lowest_img_v


def get_mean_track_widths(tracks_dict, lowest_img_v, highest_img_v, n_points=10):
    """
    Get the mean track widths at evenly spaced v-coordinates between the lowest and highest v-coordinates in the image.

    Args:
        tracks_dict (dict): Dictionary containing track information.
        lowest_img_v (float): The lowest v-coordinate in the image.
        highest_img_v (float): The highest v-coordinate in the image.
        n_points (int): Number of points to sample between lowest and highest v-coordinates.

    Returns:
        tuple: A tuple containing the sampled v-coordinates and the corresponding mean track widths.
    """

    v_samples = np.linspace(np.ceil(lowest_img_v), np.floor(highest_img_v), n_points)
    mean_track_widths = []
    v_samples_valid = []
    for v in v_samples:
        track_widths = get_all_track_widths(tracks_dict, v)
        if len(track_widths) > 0:
            mean_width = np.mean(list(track_widths.values()))
            mean_track_widths.append(float(mean_width))
            v_samples_valid.append(float(v))

    return v_samples_valid, mean_track_widths



def track_width_line_parameters(gt_rails, track_ids, right_rail, min_overlap=10, n_points=10):
    """
    Compute the parameters of a linear regression line that fits the mean track widths at different v-coordinates.

    If the maximal v-coordinate range across all tracks with both left and right rails is less than min_overlap pixels, the function will return None for both slope and y-intercept.
    
    Args:
        gt_rails (list): List of ground truth rail polylines.
        track_ids (list): List of track IDs corresponding to each rail.
        right_rail (list): List indicating whether each rail is a right rail (1) or left rail (0).
        min_overlap (int): Minimum overlap in px required for tracks to be considered valid.

    Returns:
        tuple: A tuple containing the slope (m) and y-intercept (b) of the linear regression line that fits the mean track widths at different v-coordinates.
    """
    track_dict = get_track_dict(gt_rails, track_ids, right_rail)
    highest_img_v, lowest_img_v = get_track_v_range(track_dict)
    if highest_img_v == float('-inf') or lowest_img_v == float('inf') or abs(highest_img_v - lowest_img_v) < min_overlap:
        return None, None  # No valid tracks found
    
    v_samples, mean_track_widths = get_mean_track_widths(track_dict, lowest_img_v, highest_img_v, n_points=n_points)
    if len(v_samples) < 2 or abs(v_samples[-1] - v_samples[0]) < min_overlap:
        return None, None  # Not enough valid points to fit a line
    
    m, b = np.polyfit(v_samples, mean_track_widths, 1) # Fit a linear regression line to the data

    return float(m), float(b)
