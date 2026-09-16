from Benchmarks.RAILBENCH_Rail.metrics.LineAP.line_ap import LineAP
from Benchmarks.RAILBENCH_Rail.metrics.ChamferAP.chamfer_ap import ChamferAP


def run_eval(gt, preds, metric):
    assert metric in ["LineAP", "ChamferAP"], \
        "For rail track detection, metric must be 'LineAP', or 'ChamferAP'"

    if metric == 'LineAP':

        lineAP = LineAP(
            predictions= preds, 
            gt= gt,
            sample_distance=2, 
            abs_sample_distance_flag=False,
            dist_thr_mode='track_width',
            max_detections=100
        )

        lineAP.evaluate(dist_thresholds=[3,5,7,10,20], min_dist_threshold=5, orient_threshold=10)

        lineAP.print_summary()

        return lineAP.return_results()

    else:


        chamfer_ap = ChamferAP(
                predictions=preds,
                gt=gt,
                num_sample_points=50,
                dist_thr_mode='track_width',
                max_detections=100
            )

        chamfer_ap.evaluate(dist_thresholds=[5, 10, 20, 30, 50], min_dist_threshold=10)

        chamfer_ap.print_summary()

        return chamfer_ap.return_results()

        

