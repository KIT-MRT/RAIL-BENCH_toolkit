import matplotlib.colors as mcolors
from matplotlib import colormaps
import matplotlib.pyplot as plt
import numpy as np
import cv2

from Benchmarks.RAILBENCH_Rail.metrics.LineAP.polyline_sampling import segment_sampling, point_sampling, segment_in_ignore


GT_COLOR = 'dodgerblue'
PRED_COLOR = 'orange'
TP_COLOR = 'lime'
TP2_COLOR = 'green'
F_COLOR = 'red'
F2_COLOR = 'magenta'
GREY_COLOR = 'lightgrey'
GT_COLOR_LIGHT = "#a3c9f7"
PRED_COLOR_LIGHT = "#f7d9a3"


def visualize_pointwise_evaluation(lines_labels, lines_pred, points_labels, points_pred, true_positives, matched_gt_points, figsize=(12, 12), dot_size=20):

    gt_color = 'dodgerblue'
    pred_color = 'orange'
    tp_color = 'lime'
    inc_color = 'red'
    b_color = 'lightgrey'

    TP = true_positives
    TP_GT = matched_gt_points
    # TP_pred = true_positives_predictions

    gt_patch = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=gt_color, markersize=4)
    pred_patch = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=pred_color, markersize=4)
    green_patch = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=tp_color, markersize=4)
    red_patch = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=inc_color, markersize=4)
    b_patch = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=b_color, markersize=4)

    # 1. plot: GT and predictions
    fig, ax = plt.subplots(figsize=figsize)

    draw_oriented_polylines(ax, lines_labels, color=gt_color, s_marker=5, s_arrow=15)
    draw_oriented_polylines(ax, lines_pred, color=pred_color,  s_marker=5, s_arrow=15)

    ax.legend([gt_patch, pred_patch], ['Ground Truth', 'Predictions'], loc='upper right', markerscale=2)

    plt.gca().invert_yaxis()
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

    # 2. plot: TP and FP predictions
    fig, ax = plt.subplots(figsize=figsize)

    draw_oriented_polylines(ax, lines_labels, color=gt_color, s_marker=5, s_arrow=15, zorder=-2)
    draw_oriented_polylines(ax, lines_pred, color=b_color,  s_marker=5, s_arrow=15, zorder=-1)

    ax.scatter(x=points_pred[TP, 0], y=points_pred[TP, 1], color=tp_color, s=dot_size, marker='.', zorder=1)
    ax.scatter(x=points_pred[~np.array(TP), 0], y=points_pred[~np.array(TP), 1], color=inc_color, s=dot_size, marker='.', zorder=2)

    ax.legend([gt_patch, b_patch, green_patch, red_patch], ['Ground Truth', 'Predictions', 'TP predictions', 'FP predictions'], loc='upper right', markerscale=2)

    plt.gca().invert_yaxis()
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title('Predictions: TP and FP')
    plt.show()

    # 3. plot: TP and FN GT
    fig, ax = plt.subplots(figsize=figsize)

    draw_oriented_polylines(ax, lines_labels, color=b_color, s_marker=5, s_arrow=15, zorder=-2)
    draw_oriented_polylines(ax, lines_pred, color=pred_color,  s_marker=5, s_arrow=15, zorder=-1)

    ax.scatter(x=points_labels[TP_GT, 0], y=points_labels[TP_GT, 1], color=tp_color, s=dot_size, marker='.', zorder=1)
    ax.scatter(x=points_labels[~np.array(TP_GT), 0], y=points_labels[~np.array(TP_GT), 1], color=inc_color, s=dot_size, marker='.', zorder=2)

    ax.legend([b_patch, pred_patch, green_patch, red_patch], ['Ground Truth', 'Predictions', 'TP matches', 'FN'], loc='upper right', markerscale=2)

    plt.gca().invert_yaxis()
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title('Ground truth: TP matches and FN')
    plt.show()


def visualize_segmentwise_evaluation(image, pred_segments, gt_segments, gt_points,
                                    true_positives, matched_gt_points, 
                                    add_gt_rails=False,
                                    dot_size=20, line_thickness=10, figsize=(20, 14), fontsize=15, 
                                    title='Matchings between GT points and predicted segments'):
    """
    Visualizes the segment-wise evaluation of predicted segments against ground truth segments.

    Parameters
    ----------
    image: str or np.ndarray
        Path to the image or the image itself in RGB format.
    pred_segments: list
        List of predicted segments, where each segment is a list of points [[u1, v1], [u2, v2], ...].
    gt_segments: list
        List of ground truth segments, where each segment is a list of points [[u1, v1], [u2, v2], ...].
    gt_points: list
        List of ground truth points in the format [[u1, v1], [u2, v2], ...], where each point is the midpoint of a segment in gt_segments.
    true_positives: list
        List indicating true positives for predicted segments, shape (M,) where M is the number of predicted segments.
    matched_gt_points: list
        List indicating matched ground truth points, shape (N,) where N is the number of ground truth points.
    add_gt_rails: bool
        Whether to add ground truth rails to the visualization.
    dot_size: int
        Size of the dots used to visualize points.
    line_thickness: int
        Thickness of the lines used to visualize segments.
    figsize: tuple
        Size of the figure for visualization.
    fontsize: int
        Font size for titles and labels.
    title: str
        Title of the plot.
    
    """

    assert len(pred_segments) == len(true_positives), "Length of predicted segments and true positives must match."
    assert len(gt_segments) == len(matched_gt_points), "Length of ground truth segments and matched ground truth points must match."
    assert len(gt_points) == len(gt_segments), "Each ground truth segment must have a corresponding midpoint."


    if isinstance(image, str):
        image_path = image
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        img = image.copy()

    gt_points = np.array(gt_points)

    TP = true_positives.astype(bool)
    TP_GT = matched_gt_points.astype(bool)

    matched_segments = []
    not_matched_segments = []
    for i, is_tp in enumerate(TP):
        if is_tp:
            matched_segments.append(pred_segments[i])
        else:
            not_matched_segments.append(pred_segments[i])

    img = draw_oriented_polylines2(img, matched_segments, color=TP_COLOR, thickness=line_thickness)
    img = draw_oriented_polylines2(img, not_matched_segments, color=F_COLOR, thickness=line_thickness)

    if add_gt_rails:
        img = draw_oriented_polylines2(img, gt_segments, color=GREY_COLOR, thickness=line_thickness)

    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(img)

    ax.scatter(x=gt_points[TP_GT, 0], y=gt_points[TP_GT, 1], color=TP2_COLOR, s=dot_size, marker='.', zorder=1)
    ax.scatter(x=gt_points[~np.array(TP_GT), 0], y=gt_points[~np.array(TP_GT), 1], color=F2_COLOR, s=dot_size, marker='.', zorder=2)

    ax.legend(create_color_patches([TP_COLOR, F_COLOR, TP2_COLOR, F2_COLOR]), ['TP pred', 'FP pred', 'TP gt', 'FN gt'], loc='upper right', markerscale=2)

    plt.title(title, fontsize=fontsize)
    plt.tight_layout()

    return fig



def plot_GT_pred_lines(gt_rails, pred_rails, image_path, line_thickness=5, figsize=(15, 10)):
    """
    
    Parameters
    ----------
    gt_rails: list
        list of lists of points (u, v) for GT rails
    pred_rails: list
        list of lists of points (u, v) for predicted rails
    image_path: str
        path to the image
    
    """
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_rgb_pred = img_rgb.copy()

    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    #draw_oriented_polylines(ax, gt_rails, color='dodgerblue', s_marker=5, s_arrow=15)
    img_rgb_gt = draw_oriented_polylines2(img_rgb, gt_rails, color=GT_COLOR, thickness=line_thickness)
    ax.imshow(img_rgb_gt)
    plt.title('Ground Truth')
    plt.tight_layout()
    plt.show()


    fig, ax = plt.subplots(figsize=figsize)
    #ax.imshow(img_rgb)
    #draw_oriented_polylines(ax, pred_rails, color='orange',  s_marker=5, s_arrow=15)
    img_rgb_pred = draw_oriented_polylines2(img_rgb_pred, pred_rails, color=PRED_COLOR, thickness=line_thickness)
    ax.imshow(img_rgb_pred)
    plt.title('Prediction')
    plt.tight_layout()
    plt.show()


def plot_GT_pred_pts(gt_pts, pred_pts, image_path, dot_size=5, fontsize=15):
    
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Plot
    fig, ax = plt.subplots(figsize=(15, 10))
    ax.imshow(img_rgb)
    ax.scatter(x=gt_pts[:, 0], y=gt_pts[:, 1], color='dodgerblue', s=dot_size, marker='.', zorder=1)
    plt.title('Ground Truth', fontsize=fontsize)
    plt.tight_layout()
    ax.axis('off')
    plt.show()


    fig, ax = plt.subplots(figsize=(15, 10))
    ax.imshow(img_rgb)
    ax.scatter(x=pred_pts[:, 0], y=pred_pts[:, 1], color='orange', s=dot_size, marker='.', zorder=2)
    plt.title('Prediction', fontsize=fontsize)
    plt.tight_layout()
    ax.axis('off')
    plt.show()

def visualize_rails(image, rails, title='Title', line_thickness=10, colormap='tab20', figsize=(15, 10)):

    if isinstance(image, str):
        image_path = image
        img = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        img_rgb = image.copy()

    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    #draw_oriented_polylines(ax, gt_rails, color='dodgerblue', s_marker=5, s_arrow=15)
    img_rgb_gt = draw_oriented_polylines2(img_rgb, rails, single_color=False, colormap=colormap, thickness=line_thickness)
    ax.imshow(img_rgb_gt)
    plt.title(title)
    plt.tight_layout()
    plt.show()


def draw_oriented_polylines(ax, lines, color='orange', single_color=True, s_marker=5, s_arrow=15, zorder=1):
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

    colormap = colormaps['tab20']

    for i, l in enumerate(lines):    
        if not single_color:
            color = colormap(int(i % 20))

        for pt in l:
            
            ax.scatter(pt[0], pt[1], color=color, s=s_marker, zorder=zorder)

        for i in range(len(l)-1):
            ax.annotate('', 
                xytext=l[i], # start arrow
                xy=l[i+1], # end arrow
                arrowprops=dict(arrowstyle="->", color=color),
                size=s_arrow, zorder=zorder)

    # This code is for experiments purpose for removing noisy segments.
    # for i, line in enumerate(lines):
    #     if not single_color:
    #         color = colormap(int(i % 20))
    #
    #     line = np.array(line)
    #
    #     # Draw points
    #     for pt in line:
    #         ax.scatter(pt[0], pt[1], color=color, s=s_marker, zorder=zorder)
    #
    #     # Draw arrows
    #     for j in range(0, len(line) - 1):
    #         p1 = line[j]
    #         p2 = line[j + 1]
    #
    #         # Skip degenerate or noisy segments
    #         if np.linalg.norm(p2 - p1) < 1e-3:
    #             continue
    #
    #         ax.annotate(
    #             '',
    #             xy=p2,  # end of arrow
    #             xytext=p1,  # start of arrow
    #             arrowprops=dict(arrowstyle="->", color=color),
    #             size=s_arrow,
    #             zorder=zorder
    #         )

def draw_oriented_polylines2(img, lines, color='orange', single_color=True, colormap='tab20', thickness=10):
    if single_color:
        color = (np.array(mcolors.to_rgb(color))*255).tolist()
    else:
        colormap = colormaps[colormap]

    for i, l in enumerate(lines):    
        if not single_color:
            color = colormap(int(i % 20))
            color = (np.array(mcolors.to_rgb(color))*255).tolist()

        for i in range(len(l)-1):
            #pt1 = np.array(l[i], dtype=int).tolist()
            pt1 = [int(l[i][0]), int(l[i][1])]
            pt2 = [int(l[i+1][0]), int(l[i+1][1])]
            cv2.arrowedLine(img, pt1, pt2, color, thickness=thickness, tipLength=0.2)
    return img

def create_color_patches(c_list):
    patches = []
    for c in c_list:
        patches.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=c, markersize=4))
    return patches



def visualize_graph_in_image(image, pred_segments, gt_segments, gt_points, graph,
                             dot_size=20, line_thickness=10, figsize=(20, 14), fontsize=15, 
                             title='Matchings between GT points and predicted segments'):
    
    assert len(gt_points) == len(gt_segments), "Each ground truth segment must have a corresponding midpoint."

    if isinstance(image, str):
        image_path = image
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        img = image.copy()

    # extract from graph which segments are considered in graph 
    pred_idx_g = list(set([item[0] for item in graph]))
    gt_idx_g = list(set([item[1] for item in graph]))

    # assign predicted segments into relevant and non-relevant
    rel_pred_segments = []
    non_rel_pred_segments = []
    for i, seg in enumerate(pred_segments):
        if i in pred_idx_g:
            rel_pred_segments.append(seg)
        else:
            non_rel_pred_segments.append(seg)

    img = draw_oriented_polylines2(img, rel_pred_segments, color=PRED_COLOR, thickness=line_thickness)
    img = draw_oriented_polylines2(img, non_rel_pred_segments, color=PRED_COLOR_LIGHT, thickness=line_thickness)


    gt_points = np.array(gt_points)
    gt_points_mask = np.array([i in gt_idx_g for i in range(len(gt_points))])


    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(img)

    ax.scatter(x=gt_points[gt_points_mask, 0], y=gt_points[gt_points_mask, 1], color=GT_COLOR, s=dot_size, marker='.', zorder=1)
    ax.scatter(x=gt_points[~gt_points_mask, 0], y=gt_points[~gt_points_mask, 1], color=GT_COLOR_LIGHT, s=dot_size, marker='.', zorder=2)


    ax.legend(create_color_patches([PRED_COLOR, PRED_COLOR_LIGHT, GT_COLOR, GT_COLOR_LIGHT]), ['Predictions with specific confidence score', 'Other predictions', 'Possible GT matches', 'Other GT points'], loc='upper right', markerscale=2)

    plt.title(title, fontsize=fontsize)
    plt.tight_layout()

    fig.axes[0].axis('off')

    return fig


def visualize_matched_graph_in_image(image, pred_segments, gt_segments, gt_points, 
                                     true_positives, matched_gt_points, graph,
                                     dot_size=20, line_thickness=10, figsize=(20, 14), fontsize=15, 
                                     title='Matchings between GT points and predicted segments'):
    
    assert len(gt_points) == len(gt_segments), "Each ground truth segment must have a corresponding midpoint."

    if isinstance(image, str):
        image_path = image
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        img = image.copy()

    # extract from graph which segments are considered in graph 
    pred_idx_g = list(set([item[0] for item in graph]))
    gt_idx_g = list(set([item[1] for item in graph]))

    # predictions: 
    rel_matched_pred_segments = []
    rel_not_matched_pred_segments = []
    non_rel_pred_segments = []
    for i, seg in enumerate(pred_segments):
        if i in pred_idx_g:
            if true_positives[i]:
                rel_matched_pred_segments.append(seg)
            else:
                rel_not_matched_pred_segments.append(seg)
        else:
            non_rel_pred_segments.append(seg)

    img = draw_oriented_polylines2(img, rel_not_matched_pred_segments, color=F_COLOR, thickness=line_thickness)
    img = draw_oriented_polylines2(img, rel_matched_pred_segments, color=TP_COLOR, thickness=line_thickness)
    img = draw_oriented_polylines2(img, non_rel_pred_segments, color=PRED_COLOR_LIGHT, thickness=line_thickness)

    # GT points
    gt_points = np.array(gt_points)
    gt_points_non_relevant = np.array([not(i in gt_idx_g) for i in range(len(gt_points))])
    gt_points_matched = np.array([i in gt_idx_g and matched_gt_points[i] for i in range(len(gt_points))]).astype(bool)
    gt_points_not_matched = np.array([i in gt_idx_g and not matched_gt_points[i] for i in range(len(gt_points))])


    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(img)

    ax.scatter(x=gt_points[gt_points_matched, 0], y=gt_points[gt_points_matched, 1], color=TP2_COLOR, s=dot_size, marker='.', zorder=3)
    ax.scatter(x=gt_points[gt_points_not_matched, 0], y=gt_points[gt_points_not_matched, 1], color=F2_COLOR, s=dot_size, marker='.', zorder=2)
    ax.scatter(x=gt_points[gt_points_non_relevant, 0], y=gt_points[gt_points_non_relevant, 1], color=GT_COLOR_LIGHT, s=dot_size, marker='.', zorder=1)

    #ax.legend(create_color_patches([PRED_COLOR, PRED_COLOR_LIGHT, GT_COLOR, GT_COLOR_LIGHT]), ['Predictions with specific confidence score', 'Other predictions', 'Possible GT matches', 'Other GT points'], loc='upper right', markerscale=2)

    plt.title(title, fontsize=fontsize)
    plt.tight_layout()

    fig.axes[0].axis('off')

    return fig

