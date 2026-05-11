from utils.helpers import load_json
import argparse
import os

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_path", type=str, help="Path to prediction folder")
    parser.add_argument("--is_railbench_test", action="store_true", help="Flag indicating if the folder contains predictions for RAIL-BENCH Tracking test set (default: False)")

    args = parser.parse_args()

    print("\nChecking predictions for RAIL-BENCH Tracking Challenge...")
    print("Check files in folder:", args.pred_path)
    if args.is_railbench_test:
        print("Assuming predictions are for RAIL-BENCH Tracking test set.")
        gt_infos = load_json('format/railbench_tracking_test_image_infos.json')

    files = os.listdir(args.pred_path)
    json_files = [f for f in files if f.endswith('.json')]
    if args.is_railbench_test:
        expected_json_files = {'scene_31_test.json', 'scene_32_test.json', 'scene_33_test.json', 'scene_34_test.json'}
        assert set(json_files) == expected_json_files, f"Expected JSON files {expected_json_files}, but found {set(json_files)}."
    else:
        assert len(json_files) > 0, "No JSON files found in the prediction folder."

    for json_file in json_files:
        json_path = os.path.join(args.pred_path, json_file)
        prediction_data = load_json(json_path)

        assert isinstance(prediction_data, list), f"Prediction file {json_file} does not contain a list."
        for i, frame in enumerate(prediction_data):
            assert "name" in frame and "index" in frame and "labels" in frame, f"Frame in {json_file} is missing required keys."
            
            if args.is_railbench_test:
                # Check if frame name corresponds to expected format for RAIL-BENCH Tracking test set
                assert frame['name'] == gt_infos[json_file][i], f"Frame name {frame['name']} in {json_file} does not match expected name {gt_infos[json_file][i]}."
                assert frame['index'] == i, f"Frame index {frame['index']} in {json_file} does not match expected index {i}."

            for label in frame['labels']:
                assert "id" in label and "category" in label and "bbox" in label, f"Label in {json_file} is missing required keys."
                assert isinstance(label['category'], str), f"Label category {label['category']} in {json_file} is not a string."
                assert isinstance(label['bbox'], list) and len(label['bbox']) == 4, f"Label bbox {label['bbox']} in {json_file} is not a list of 4 elements."
                assert all(isinstance(v, (int, float)) for v in label['bbox']), f"Label bbox {label['bbox']} in {json_file} contains invalid values."

                if args.is_railbench_test:
                    assert label['category'] == 'person'

    print("\nAll checks passed!\n")

