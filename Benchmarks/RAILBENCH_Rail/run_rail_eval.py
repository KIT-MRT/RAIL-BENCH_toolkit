import os
import re

from utils.helpers import load_json, save_json
from Benchmarks.RAILBENCH_Rail.rail_metrics import run_eval
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    
    parser.add_argument("--metric", choices=['ChamferAP', 'LineAP'], help="Metric")
    parser.add_argument("--project", type=str, help="Experiment name")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing results")
    args = parser.parse_args()

    metric = args.metric
    project_dir = os.path.join("data", args.project)
    overwrite = args.overwrite

    if overwrite:
        print("Overwriting existing results...")
    else:
        print("Not overwriting existing results. Existing results will be skipped.")

    annotation_files = [f for f in os.listdir(os.path.join(project_dir, "annotations")) if f.endswith(".json")]

    for ann_file in annotation_files:
        split = re.match(r".*_(.*)\.json", ann_file).group(1)
        gt_path = os.path.join(project_dir, "annotations", ann_file)
        gt = load_json(gt_path)

        for detector in os.listdir(os.path.join(project_dir, "detectors")):
            print("-----")
            print(f"Evaluating {detector} on {split} split...")
            detector_path = os.path.join(project_dir, "detectors", detector)

            if not os.path.isdir(detector_path):
                continue

            for pred_file in os.listdir(detector_path):
                if re.match(r".*_(.*)\.json", pred_file).group(1) == split:
                    save_path = os.path.join(project_dir, "results", metric, detector)
                    os.makedirs(save_path, exist_ok=True)
                    save_file = os.path.join(save_path, f"eval_{split}.json")
                    print(f"Saving results to {save_file} ...")
                    if os.path.exists(save_file) and not overwrite:
                        print(f"Results for {detector} on {split} split already exist. Skipping...")
                        continue
                
                    dt_path = os.path.join(detector_path, pred_file)
                    dt = load_json(dt_path)

                    results = run_eval(gt, dt, metric=metric)

                    save_json(results, save_file)
