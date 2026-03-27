import os

from Benchmarks.RAILBENCH_Rail.metrics.LineAP.line_ap_metric import LineAP
from Benchmarks.RAILBENCH_Rail.metrics.LineAP.visualizations import visualize_segmentwise_evaluation
from Benchmarks.RAILBENCH_Rail.viz.image import image_preparation


def get_viz_output(img_ident, preds, gt, 
                   sample_distance=50, abs_sample_distance_flag=True,
                   dist_thres=10, use_abs_dist=True,
                   orient_thres=10):
    
    lineap = LineAP(predictions=preds, gt=gt, 
                    sample_distance=sample_distance, abs_sample_distance_flag=abs_sample_distance_flag, 
                    extended_summary=True)

    if use_abs_dist:
        dist_thres = dist_thres
    else:
        dist_thres = int(dist_thres/100 * lineap.gt[img_ident]['image_width'])

    gt_rails = lineap.gt[img_ident]['rails'].copy()
    ignore_areas = lineap.gt[img_ident]['ignore_areas'].copy()
    pred_rails = lineap.predictions[img_ident]['rails'].copy()
    pred_confidence = lineap.predictions[img_ident]['score'].copy()

    output = lineap.compute_tp_fp_single_frame(pred_rails=pred_rails, pred_confidence=pred_confidence,
                                               gt_rails=gt_rails, ignore_areas=ignore_areas, 
                                               sample_distance=lineap.gt[img_ident]['sample_distance'], 
                                               dist_thres=dist_thres, orient_thres=orient_thres)

    return {
        "is_tp": output['true_positives'],
        "pred_lanes": output['pred_segments'],
        "gt_lanes": output['gt_segments'],
    }



def detailed_viz(img_ident, image_dir, preds, gt, 
                   sample_distance=50, rel_sample_distance=0.02, use_abs_sample_distance=True,
                   dist_thres=10, rel_dist_thres=0.005, use_abs_dist=True, orient_thres=10,
                   dot_size=30, line_thickness=2):
    
    lineap = LineAP(predictions=preds, gt=gt, 
                    sample_distance=sample_distance, rel_sample_distance=rel_sample_distance, use_abs_sample_distance=use_abs_sample_distance, 
                    extended_summary=True)

    if use_abs_dist:
        dist_thres = dist_thres
    else:
        dist_thres = int(rel_dist_thres * lineap.gt[img_ident]['image_width'])

    gt_rails = lineap.gt[img_ident]['rails']
    ignore_areas = lineap.gt[img_ident]['ignore_areas']
    pred_rails = lineap.predictions[img_ident]['rails']
    pred_confidence = lineap.predictions[img_ident]['score'].copy()

    output = lineap.compute_tp_fp_single_frame(pred_rails=pred_rails, pred_confidence=pred_confidence,
                                               gt_rails=gt_rails, ignore_areas=ignore_areas, 
                                               sample_distance=lineap.gt[img_ident]['sample_distance'], 
                                               dist_thres=dist_thres, orient_thres=orient_thres)
    
    img = image_preparation(os.path.join(image_dir, img_ident))

    fig = visualize_segmentwise_evaluation(
        image=img,
        pred_segments=output['pred_segments'],
        gt_segments=output['gt_segments'],
        gt_points=output['gt_points'],
        add_gt_rails=False,
        true_positives=output['true_positives'],
        matched_gt_points=output['matched_gt_points'],
        figsize=(15, 8), fontsize=12, dot_size=dot_size, line_thickness=line_thickness,
        title=f"Matchings for {img_ident} with rel_dist_thres={rel_dist_thres}"
    )

    fig.axes[0].axis('off')

    return fig