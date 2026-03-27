import os
from utils.helpers import save_json
from railbench_vegetation import evaluate_vegetation_segmentation

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate vegetation segmentation")
    parser.add_argument("--split", "-s", default="test", help="Dataset split to evaluate (e.g. train, val, test)")
    parser.add_argument("--pred_path", "-p", help="Path to directory with predictions")
    parser.add_argument("--gt_path", "-g", help="Path to directory with ground truth masks")
    parser.add_argument("--expected_num_gt_files", type=int, default=None, help="Expected number of ground truth files (optional, used for sanity check)")
    parser.add_argument("--project_name", default='eval_results', help="Name of the project (used for saving results)")
    parser.add_argument("--overwrite", action='store_true', help="Whether to overwrite existing results (default: False)")
    args = parser.parse_args()
    split = args.split
    pred_path = os.path.join(args.pred_path, split)
    gt_path = os.path.join(args.gt_path, split)

    output_path = f"results/{args.project_name}"
    os.makedirs(output_path, exist_ok=True)
    output_file = os.path.join(output_path, f"scores_{split}.json")
    if os.path.exists(output_file):
        if args.overwrite:
            print(f"Output file {output_file} already exists. Overwriting...")
        else:
            print(f"Output file {output_file} already exists. Please choose a different name for the experiment or use the --overwrite flag.")
            exit(1)

    scores = evaluate_vegetation_segmentation(gt_path, pred_path, split=split, n_gt_files_expected=args.expected_num_gt_files)

    print("\n" + "="*60)
    print("EVALUATION SCORES")
    print("="*60)
    
    # IoU Metrics
    print("\nIoU Metrics:")
    print("-" * 60)
    print(f"  Mean IoU (all classes):           {scores['iou_mean']:.4f}")
    print(f"  Mean IoU (vegetation classes):    {scores['iou_vegetation_mean']:.4f}")
    print(f"  IoU (single vegetation class):    {scores['iou_single_vegetation_class']:.4f}")
    print(f"  IoU (background):                 {scores['iou_background']:.4f}")
    print(f"  IoU (vegetation_low):             {scores['iou_vegetation_low']:.4f}")
    print(f"  IoU (vegetation_high):            {scores['iou_vegetation_high']:.4f}")
    
    # Accuracy Metrics
    print("\nAccuracy Metrics:")
    print("-" * 60)
    print(f"  Mean Accuracy (all classes):          {scores['accuracy_mean']:.4f}")
    print(f"  Mean Accuracy (vegetation classes):   {scores['accuracy_vegetation_mean']:.4f}")
    print(f"  Accuracy (single vegetation class):   {scores['accuracy_single_vegetation_class']:.4f}")
    print(f"  Accuracy (background):                {scores['accuracy_background']:.4f}")
    print(f"  Accuracy (vegetation_low):            {scores['accuracy_vegetation_low']:.4f}")
    print(f"  Accuracy (vegetation_high):           {scores['accuracy_vegetation_high']:.4f}")
    
    print("="*60)
    print(f"\nSaving scores to {output_file}")
    save_json(scores, output_file)