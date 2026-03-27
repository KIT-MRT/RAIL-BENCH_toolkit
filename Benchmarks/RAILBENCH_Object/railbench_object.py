from Benchmarks.RAILBENCH_Object.metrics.eval_objects import run_eval_objects

def compute_scores(gt, preds):
    print("Compute mean average precision (mAP) for object detection...")
    results = run_eval_objects(gt, preds, gt_box_format="xywh", dt_box_format="xywh")
    return results