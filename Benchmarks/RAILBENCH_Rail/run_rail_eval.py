import os
import re

from utils.helpers import load_json, save_json
from Benchmarks.RAILBENCH_Rail.rail_metrics import run_eval
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    
    parser.add_argument("--metric", choices=['ChamferAP', 'LineAP', 'RailBench'], help="Metric")
    parser.add_argument("--project", type=str, help="Experiment name")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing results")
    args = parser.parse_args()

    project_dir = os.path.join("data", args.project)
    overwrite = args.overwrite

    if overwrite:
        print("Overwriting existing results...")
    else:
        print("Not overwriting existing results. Existing results will be skipped.")

    annotation_files = [f for f in os.listdir(os.path.join(project_dir, "annotations")) if f.endswith(".json")]

    metric = args.metric

    if metric == "RailBench":
        metrics = ["LineAP", "ChamferAP"]
    else:
        metrics = [metric]
    
    for m in metrics:
        print(f"\nEvaluating metric: {m} ...\n")
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
                        save_path = os.path.join(project_dir, "results", m, detector)
                        os.makedirs(save_path, exist_ok=True)
                        save_file = os.path.join(save_path, f"eval_{split}.json")
                        print(f"Saving results to {save_file} ...")
                        if os.path.exists(save_file) and not overwrite:
                            print(f"Results for {detector} on {split} split already exist. Skipping...")
                            continue
                    
                        dt_path = os.path.join(detector_path, pred_file)
                        dt = load_json(dt_path)

                        results = run_eval(gt, dt, metric=m)

                        save_json(results, save_file)


    if metric == "RailBench":
        print("\nComputing RailBench mAP across both metrics...\n")
        # Compute mAP across both metrics
        for ann_file in annotation_files:
            split = re.match(r".*_(.*)\.json", ann_file).group(1)
            save_path = os.path.join(project_dir, "results", "RailBench")
            os.makedirs(save_path, exist_ok=True)

            for detector in os.listdir(os.path.join(project_dir, "detectors")):
                print("-----")
                print(f"Evaluating {detector} on {split} split...")
                detector_path = os.path.join(project_dir, "detectors", detector)

                if not os.path.isdir(detector_path):
                    continue

                for pred_file in os.listdir(detector_path):
                    if re.match(r".*_(.*)\.json", pred_file).group(1) == split:
                        save_path = os.path.join(project_dir, "results", "RailBench", detector)
                        os.makedirs(save_path, exist_ok=True)
                        save_file = os.path.join(save_path, f"eval_{split}.json")
                        print(f"Saving results to {save_file} ...")
                        if os.path.exists(save_file) and not overwrite:
                            print(f"Results for {detector} on {split} split already exist. Skipping...")
                            continue

                lineap_file = os.path.join(project_dir, "results", "LineAP", detector, f"eval_{split}.json")
                chamferap_file = os.path.join(project_dir, "results", "ChamferAP", detector, f"eval_{split}.json")

                lineap_results = load_json(lineap_file)
                chamferap_results = load_json(chamferap_file)

                mAP = (lineap_results["mAP"] + chamferap_results["mAP"]) / 2

                railbench_results = {
                    "LineAP": lineap_results,
                    "ChamferAP": chamferap_results,
                    "mAP": mAP
                }

                save_json(railbench_results, save_file)
