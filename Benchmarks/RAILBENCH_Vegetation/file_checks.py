import re
import os

def check_file_naming(pred_path, gt_path, split, n_gt_files_expected=None):
    if not os.path.exists(pred_path):
        print(f"Prediction path {pred_path} does not exist.")
        exit(1)
    if not os.path.exists(gt_path):
        print(f"Ground truth path {gt_path} does not exist.")
        exit(1)

    gt_file_pattern = re.compile(rf"mask_(\d+)_{split}\.png")
    pred_file_pattern = re.compile(rf"mask_pred_(\d+)_{split}\.png")

    gt_files = [f for f in os.listdir(gt_path) if f.endswith('.png')]
    gt_files = sorted(gt_files, key = lambda file_name: re.match(gt_file_pattern, file_name).group(1))
    gt_ids = [re.match(gt_file_pattern, f).group(1) for f in gt_files]

    if n_gt_files_expected is not None and n_gt_files_expected > 0:
        if len(gt_ids) != n_gt_files_expected:
            raise ValueError(f"Expected {n_gt_files_expected} ground truth files, but found {len(gt_ids)}. Please check the gt_path and split name.")

    all_pred_files = [f for f in os.listdir(pred_path) if f.endswith('.png')]
    pred_ids = [re.match(pred_file_pattern, f).group(1) for f in all_pred_files]
    pred_files = []
    for id in gt_ids:
        if id in pred_ids:
            pred_files.append(all_pred_files[pred_ids.index(id)])
        else:
            raise ValueError(f"No prediction file found for ground truth id {id}")
        
    gt_files = [os.path.join(gt_path, f) for f in gt_files]
    pred_files = [os.path.join(pred_path, f) for f in pred_files]

    return gt_files, pred_files

