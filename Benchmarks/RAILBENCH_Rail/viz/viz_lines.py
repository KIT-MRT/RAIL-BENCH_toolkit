import numpy as np
import cv2

from utils.viz.colors import BRIGHT_COLORS_RGB


def rb_anns_preparation(annotations, image_id=1):
    """
    Takes railbench rail annotations and extracts rails and ignore areas for a given image_id.

    Args:
    ------
    annotations: dict containing the annotations loaded from the railbench json file
    image_id: id of the image for which to extract the rails and ignore areas
    """

    rails = []
    ignore_areas = []

    assert annotations['categories'][0]['id'] == 1 and annotations['categories'][0]['name'] == 'rail', "Expected category id 1 to be 'rail'"
    assert annotations['categories'][1]['id'] == 2 and annotations['categories'][1]['name'] == 'ignore_area', "Expected category id 2 to be 'ignore area'"

    for ann in annotations['annotations']:
        if ann['image_id'] == image_id:
            if ann['category_id'] == 1: # rail
                rails.append(ann['polyline'])
            elif ann['category_id'] == 2: # ignore area
                ignore_areas.append(ann['polygon'])

    return rails, ignore_areas

#-------------------------------------------------------------------

def visualize_tracks(img, rails, track_ids=None,
                     ignore_areas=None, add_ignore_areas_flag=True,
                     color_mode = 'instance',
                     color_rail = (255, 0, 238),
                     color_ignore_area = (51, 255, 255),
                     thickness=5, 
                     plot_arrows = False,
                     plot_keypoints = False):
    """
    Add rails and ignore areas to the image. Ignore areas can be added as semi-transparent overlays.

    There are three modes for coloring the rails: 'instance', 'single' and 'track'.
    If 'instance', each rail gets a different color (randomly assigned from a predefined list of bright colors).
    If 'single', all rails get the same color specified by color_rail.
    If 'track', rails are colored according to their track id (requires track_ids to be provided).

    Args:
    -------
    img: input image (numpy array) in RGB format
    rails: list of rails, where each rail is a list of [u, v] coordinates, i.e. [ [[u1, v1], [u2, v2], ...], [...], ... ]
    track_ids: list of track ids corresponding to the rails (optional)
    ignore_areas: list of ignore areas, where each area is a list of [u, v] coordinates, i.e. [ [[u1, v1], [u2, v2], ...], [...], ... ]
    add_ignore_areas_flag: flag to indicate whether to add ignore areas to the image
    color_mode: mode for coloring rails ('instance', 'single', or 'track')
    color_rail: color of the rails in BGR format (default magenta)
    color_ignore_area: color of the ignore areas in BGR format (default cyan)
    thickness: thickness of the rail lines
    plot_arrows: flag to indicate whether to display each polyline as arrows between consecutive anchor points 
    plot_keypoints: flag to indicate whether to plot anchor points of polylines 

    returns:
    -------
    img: output image (numpy array) in RGB format
    """

    if track_ids is not None:
        assert len(track_ids) == len(rails), "Length of track_ids must match length of rails"
        assert all(isinstance(tid, int) for tid in track_ids), "All elements in track_ids must be integers"
    assert color_mode in ['instance', 'single', 'track'], "color_mode must be either 'instance', 'single' or 'track'"
    if color_mode == 'track':
        assert track_ids is not None, "track_ids must be provided when color_mode is 'track'"

    # add ignore areas 
    if ignore_areas is not None and add_ignore_areas_flag:
        overlay = img.copy()
        alpha = 0.1
        for area in ignore_areas:
            pts = np.array(area).astype(np.int32)
            cv2.fillPoly(overlay, [pts], color=color_ignore_area) 
            cv2.polylines(img, [pts], isClosed=True, color=color_ignore_area, thickness=1)
        img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

    # add rails
    for i, rail in enumerate(rails):
        pts = np.array(rail).astype(np.int32)
        if color_mode == 'instance':
            c = BRIGHT_COLORS_RGB[i % len(BRIGHT_COLORS_RGB)]
        elif color_mode == 'track':
            c = BRIGHT_COLORS_RGB[track_ids[i] % len(BRIGHT_COLORS_RGB)]
        else:
            c = color_rail

        if not plot_arrows:
            cv2.polylines(img, [pts], isClosed=False, color=c, thickness=thickness)
            if plot_keypoints:
                for p in np.squeeze(pts):
                    cv2.circle(img, p, 12, c, thickness=-1)

        else:
            for k in range(len(rail)-1):
                pt1 = [int(rail[k][0]), int(rail[k][1])]
                pt2 = [int(rail[k+1][0]), int(rail[k+1][1])]
                cv2.arrowedLine(img, pt1, pt2, c, thickness=thickness, tipLength=0.2)

    return img



