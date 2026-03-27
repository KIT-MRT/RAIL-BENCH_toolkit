import numpy as np

from Benchmarks.RAILBENCH_Object.metrics.bounding_box import BoundingBox, get_bb_format
from Benchmarks.RAILBENCH_Object.metrics.enumerators import BBType
from Benchmarks.RAILBENCH_Object.metrics.rb_evaluator import get_coco_summary


def run_eval_objects(gt, dt, gt_box_format, dt_box_format):
    """
    Run evaluation on the given ground truth and detection dictionaries.

    Args:
        gt (dict): Ground truth dictionary.
        dt (list): Detection list.

    The ground truth dictionary should follow the RailBench or COCO format, containing:
        - 'images': List of images with 'id', 'file_name', 'width', 'height'.
        - 'annotations': List of annotations with 'id', 'image_id', 'category_id', 'bbox', 
                         'occlusion', 'iscrowd', and 'ignore'.
        - 'categories': List of categories with 'id', 'name'.

    The detections should be a list of dictionaries, each containing:
        - 'image_id': ID of the image.
        - 'category_id': ID of the category.
        - 'bbox': Bounding box coordinates.
        - 'score': Confidence score of the detection.

    Returns:
        dict: Evaluation results.
    """
    
    id_to_filename = {img['id']: img['file_name'] for img in gt['images']}

    gt_bbox_list = []
    for ann in gt['annotations']:
        BB = BoundingBox(
            image_name = id_to_filename[ann['image_id']],
            class_id = ann['category_id'],
            coordinates = ann['bbox'],
            is_crowd=ann['iscrowd'],
            occlusion=ann['occlusion'],
            ignore=ann['ignore'],

            bb_type=BBType.GROUND_TRUTH,
            format=get_bb_format(gt_box_format),
        )
        gt_bbox_list.append(BB)

    pred_bbox_list = []
    for pred in dt:
        BB = BoundingBox(
            image_name = pred['file_name'],
            class_id = pred['category_id'],
            coordinates = pred['bbox'],
            confidence=pred['score'],

            bb_type=BBType.DETECTED,
            format=get_bb_format(dt_box_format),
        )
        pred_bbox_list.append(BB)

    summary = get_coco_summary(gt_bbox_list, pred_bbox_list, categories=gt['categories'])
    return summary