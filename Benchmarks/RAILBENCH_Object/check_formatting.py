from utils.helpers import load_json
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_file", type=str, help="Path to prediction file")
    parser.add_argument("--is_railbench_test", action="store_true", help="Flag indicating if the file contains predictions for RAIL-BENCH Object test set (default: False)")

    args = parser.parse_args()

    print("\nCheck formatting of prediction file:", args.pred_file)
    if args.is_railbench_test:
        print("Assuming predictions are for RAIL-BENCH Object test set.")

    predictions = load_json(args.pred_file)

    if args.is_railbench_test:
        gt_infos = load_json('format/railbench_rail_test_image_infos.json')
        gt_images = [image['file_name'] for image in gt_infos]

    # ---------- Check formatting of prediction file ----------

    assert isinstance(predictions, list), "Prediction file must be a list"
    
    for item in predictions: 
        assert isinstance(item, dict), "Each item in the prediction list must be a dictionary"

        for k in ['file_name', 'category_id', 'bbox', 'score']:
            assert k in item, f"{k} missing"

            if k == 'file_name':
                assert isinstance(item[k], str), "file_name must be a string"
                if args.is_railbench_test:
                    assert item[k] in gt_images, f"file_name {item[k]} not found in RAIL-BENCH Object test set image IDs"

            if k == 'category_id':
                assert item[k] in list(range(1,8)), "category_id must be one of the following values: 1,2,3,4,5,6, or 7"

            if k == 'bbox':
                assert isinstance(item[k], list) and len(item[k]) == 4, "bounding box must be a list of 4 values"
                assert all(isinstance(v, (int, float)) for v in item[k]), "all bbox values must be int or float"
                
            if k == 'score':
                assert isinstance(item[k], (int, float)), "score must be a number"
                assert 0 <= item[k] <= 1, "score must be between 0 and 1"

    print("\nAll checks passed! Prediction file is correctly formatted.\n")