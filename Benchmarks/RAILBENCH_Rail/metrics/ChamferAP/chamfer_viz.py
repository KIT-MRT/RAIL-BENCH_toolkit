from Benchmarks.RAILBENCH_Rail.metrics.ChamferAP.chamfer_ap import ChamferAP

def get_viz_output(img_ident, preds, gt, num_sample_points=50, threshold=100, use_abs_dist=True):
    chamfer_ap = ChamferAP(predictions=preds, gt=gt, num_sample_points=num_sample_points, extended_summary=True)

    if use_abs_dist:
        threshold = threshold
    else:
        threshold = int(threshold/100 * chamfer_ap.gt[img_ident]['image_width'])
        
    gt_rails = chamfer_ap.gt[img_ident]['rails'].copy()
    ignore_areas = chamfer_ap.gt[img_ident]['ignore_areas'].copy()
    pred_rails = chamfer_ap.predictions[img_ident]['rails'].copy()
    pred_scores = chamfer_ap.predictions[img_ident]['score'].copy()

    output = chamfer_ap._compute_tp_fp_single_image(pred_rails=pred_rails, pred_scores=pred_scores,
                                                   gt_rails=gt_rails, threshold=threshold)

    return {
        "is_tp": output['tp_flags'],
        "pred_lanes": output['pred_rails_sorted'],
        "gt_lanes": gt_rails,
    }