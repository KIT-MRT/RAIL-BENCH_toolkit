import os 
import numpy as np
import cv2
import matplotlib.pyplot as plt

import matplotlib.colors as mcolors
from matplotlib import colormaps

from Benchmarks.RAILBENCH_Rail.viz.colors import *

def draw_polyline(img, lane, color, thickness=4, dashed=False, dash_len=20):
    """Draw a polyline onto img in-place. Optionally dashed."""
    if isinstance(color, str):
        color = hex_to_rgb(color)

    pts = [(int(round(u)), int(round(v))) for u, v in lane]
    if dashed:
        for p1, p2 in zip(pts[:-1], pts[1:]):
            # Subdivide each segment into dash / gap pairs
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            seg_len = max(1, int(np.hypot(dx, dy)))
            steps = seg_len // dash_len
            for k in range(steps):
                if k % 2 == 0:          # even → draw, odd → skip
                    t0, t1 = k / steps, (k + 1) / steps
                    a = (int(p1[0] + t0 * dx), int(p1[1] + t0 * dy))
                    b = (int(p1[0] + t1 * dx), int(p1[1] + t1 * dy))
                    cv2.line(img, a, b, color, thickness, cv2.LINE_AA)
    else:
        pts = np.array(pts).astype(np.int32)
        cv2.polylines(img, [pts], isClosed=False, color=color, thickness=thickness)


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

    for i, rail in enumerate(rails):
        pts = np.array(rail).astype(np.int32)
        if instance_coloring:
            c = BRIGHT_COLORS_BGR[i % len(BRIGHT_COLORS_BGR)]
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

def draw_polyline1(img, lines, color=(255,   0, 128), single_color=True, thickness=10, add_dots=True):
    for i, l in enumerate(lines):    
        if not single_color:
            color = BRIGHT_COLORS_BGR[i % len(BRIGHT_COLORS_BGR)]

        cv2.polylines(img, [np.array(l, dtype=int)], isClosed=False, color=color, thickness=thickness)

        if add_dots:
            for pt in l:
                cv2.circle(
                    img,
                    center=np.array(pt, dtype=int).tolist(),
                    radius=10,
                    color=(255, 0, 0),  # red dots
                    thickness=-1  # filled
                )
    return img

def add_ignore_areas(img, ignore_areas, color=(0, 0, 255), alpha=0.2):
    overlay = img.copy()
    for area in ignore_areas:
        cv2.fillPoly(overlay, [np.array(area, dtype=int)], color)
    return cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

def draw_oriented_polylines(ax, lines, color, s_marker=5, s_arrow=15, zorder=1):
    """
    Draws oriented polylines on the given axis.
    Sets points and arrows. 

    Args:
    ------
    ax: matplotlib axis

    lines: list of lists of points (u, v)

    color: color of the lines

    s_marker: size of the points

    s_arrow: size of the arrows

    """

    color = [v / 255 for v in color] 


    for i, l in enumerate(lines):    

        for pt in l:
            
            ax.scatter(pt[0], pt[1], color=color, s=s_marker, zorder=zorder)

        for i in range(len(l)-1):
            ax.annotate('', 
                xytext=l[i], # start arrow
                xy=l[i+1], # end arrow
                arrowprops=dict(arrowstyle="->", color=color),
                size=s_arrow, zorder=zorder)


def railbench_preparation(annotations, image_id=1):

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

