from file_checks import check_file_naming
from iou_scores import compute_scores


def evaluate_vegetation_segmentation(gt_path, pred_path, split='test', n_gt_files_expected=None):

    gt_files, pred_files = check_file_naming(pred_path, gt_path, split=split, n_gt_files_expected=n_gt_files_expected)
    scores = compute_scores(gt_files, pred_files)

    return scores