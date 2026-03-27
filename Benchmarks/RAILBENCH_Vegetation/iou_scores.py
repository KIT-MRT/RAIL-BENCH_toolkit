import numpy as np
from PIL import Image
import tqdm


CLASSES = ['background', 'vegetation_low', 'vegetation_high']
NUM_CLASSES = len(CLASSES)

def compute_confusion_matrix(gt, pred, num_classes):
    mask = (gt >= 0) & (gt < num_classes)
    return np.bincount(
        num_classes * gt[mask].astype(int) + pred[mask].astype(int),
        minlength=num_classes ** 2
    ).reshape(num_classes, num_classes)

def compute_scores(gt_files, pred_files):
    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    cm_veg = np.zeros((2, 2), dtype=np.int64)

    for gt_file, pred_file in tqdm.tqdm(zip(gt_files, pred_files), total=len(gt_files)):
        gt_mask = np.array(Image.open(gt_file))
        pred_mask = np.array(Image.open(pred_file))

        cm += compute_confusion_matrix(gt_mask, pred_mask, NUM_CLASSES)

        gt_veg = (gt_mask > 0).astype(np.uint8)
        pred_veg = (pred_mask > 0).astype(np.uint8)
        cm_veg += compute_confusion_matrix(gt_veg, pred_veg, 2)

    # IoU and accuracy per class
    intersection = np.diag(cm)
    union = cm.sum(axis=1) + cm.sum(axis=0) - intersection
    per_class_iou = intersection / np.maximum(union, 1)
    per_class_acc = intersection / np.maximum(cm.sum(axis=1), 1)

    # IoU and accuracy for combined vegetation class
    intersection_veg = np.diag(cm_veg)
    union_veg = cm_veg.sum(axis=1) + cm_veg.sum(axis=0) - intersection_veg
    per_class_iou_veg = intersection_veg / np.maximum(union_veg, 1)
    per_class_acc_veg = intersection_veg / np.maximum(cm_veg.sum(axis=1), 1)

    scores = {}
    # Add per-class scores
    for i, c in enumerate(CLASSES):
        scores[f"accuracy_{c}"] = per_class_acc[i].item()
        scores[f"iou_{c}"] = per_class_iou[i].item()

    # Add combined vegetation scores
    scores["accuracy_single_vegetation_class"] = per_class_acc_veg[1].item()
    scores["iou_single_vegetation_class"] = per_class_iou_veg[1].item()

    # Mean of IoU/Accuracy for vegetation classes (excluding background)
    scores["iou_vegetation_mean"] = ((per_class_iou[1] + per_class_iou[2]) / 2.0).item()
    scores["accuracy_vegetation_mean"] = ((per_class_acc[1] + per_class_acc[2]) / 2.0).item()

    # Averaged across all classes (including background)
    scores["iou_mean"] = per_class_iou.mean().item()
    scores["accuracy_mean"] = per_class_acc.mean().item()

    return scores