from Benchmarks.RAILBENCH_Rail.metrics.LineAP.line_ap_metric import LineAP
from Benchmarks.RAILBENCH_Rail.metrics.ChamferAP.chamfer_ap import ChamferAP
from Benchmarks.RAILBENCH_Rail.utils.ignore_areas import process_predictions


def run_eval(gt, preds, metric):
    assert metric in ["LineAP", "ChamferAP"], \
        "For rail track detection, metric must be 'LineAP', or 'ChamferAP'"

    if metric == 'LineAP':

        lineAP = LineAP(
            predictions= preds, 
            gt= gt,
            sample_distance=2, 
            abs_sample_distance_flag=False
        )

        lineAP.evaluate(dist_thresholds=[0.1, 0.5, 1.0], use_abs_dist=False, orient_threshold=10)

        lineAP.print_summary()

        return lineAP.return_results()

    else:
        # remove predictions in ignore areas
        preds = process_predictions(preds, gt)
        
        chamfer_ap = ChamferAP(
            predictions=preds,
            gt=gt,
            num_sample_points=50,
        )

        chamfer_ap.evaluate(chamfer_thresholds=[0.5, 1.0, 5.0], use_abs_chamfer=False)

        chamfer_ap.print_summary()

        return chamfer_ap.return_results()

        

