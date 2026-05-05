import os 
import numpy as np
import cv2
import matplotlib.pyplot as plt

import matplotlib.colors as mcolors
from matplotlib import colormaps

from utils.viz.colors import BRIGHT_COLORS_RGB, hex_to_rgb


def railbench_preparation(annotations, image_id=1):
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

def visualize_tracks(img, rails, 
                     ignore_areas=None, add_ignore_areas_flag=True,
                     color_rail = (255, 0, 238),
                     color_ignore_area = (51, 255, 255),
                     instance_coloring=False, thickness=5, 
                     plot_arrows = False,
                     plot_keypoints = False):

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
        if instance_coloring:
            c = BRIGHT_COLORS_RGB[i % len(BRIGHT_COLORS_RGB)]
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



