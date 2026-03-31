from utils.helpers import load_json
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_file", type=str, help="Path to prediction file")
    parser.add_argument("--is_railbench_test", action="store_true", help="Flag indicating if the file contains predictions for RAIL-BENCH Rail test set (default: False)")

    args = parser.parse_args()

    print("\nCheck formatting of prediction file:", args.pred_file)
    if args.is_railbench_test:
        print("Assuming predictions are for RAIL-BENCH Rail test set, will check if image IDs and rail coordinates are valid based on the test set image infos.")
    else:
        print("Not assuming predictions are for RAIL-BENCH Rail test set, will only check general formatting of prediction file without checking image IDs or rail coordinates.")

    predictions = load_json(args.pred_file)

    if args.is_railbench_test:
        gt_infos = load_json('format/railbench_rail_test_image_infos.json')
        gt_images = [image['file_name'] for image in gt_infos]
        gt_image_shapes = {image['file_name']: (image['width'], image['height']) for image in gt_infos}


    # ---------- Check formatting of prediction file ----------

    # 1. Check if json is dictionary 
    assert isinstance(predictions, dict), "Prediction file should be a dictionary."

    # 2. If is_railbench_test is True, check whether keys correspond to the images in RAIL-BENCH Rail test set
    if args.is_railbench_test:
        for image_id in predictions.keys():
            assert image_id in gt_images, f"Image ID {image_id} not found in RAIL-BENCH Rail test set."

        assert len(predictions) == len(gt_infos), f"Number of predictions ({len(predictions)}) does not match number of images in RAIL-BENCH Rail test set ({len(gt_infos)})."

    # 3. Check if entries for each image are in the correct format: 
    for image_id, pred in predictions.items():
        assert isinstance(pred, dict), f"Predictions should be a dictionary, check image {image_id}."
        assert 'rails' in pred and 'score' in pred, f"Predictions should contain 'rails' and 'score' keys, check image {image_id}."
        
        rails = pred['rails']
        scores = pred['score']

        assert isinstance(rails, list), f"'rails' should be a list, check image {image_id}."
        assert isinstance(scores, list), f"'score' should be a list, check image {image_id}."
        assert len(rails) == len(scores), f"Number of 'rails' and 'score' entries should be the same, check image {image_id}."

        for rail in rails:
            assert isinstance(rail, list), f"Each rail should be a list of points, check image {image_id}."
            assert len(rail) >= 2, f"Each rail should have at least 2 points, check image {image_id}."
            for point in rail:
                assert isinstance(point, list) and len(point) == 2, f"Each point in rail should be a list of two coordinates [u, v], check image {image_id}."
                u, v = point
                assert isinstance(u, (int, float)) and isinstance(v, (int, float)), f"Coordinates of points in rail should be numbers, check image {image_id}."

                if args.is_railbench_test:
                    # Check if rails are within the image dimensions
                    width = gt_image_shapes[image_id][0]
                    height = gt_image_shapes[image_id][1]
                    assert 0 <= u <= width, f"u coordinate {u} in rail should be between 0 and {width}, check image {image_id}."
                    assert 0 <= v <= height, f"v coordinate {v} in rail should be between 0 and {height}, check image {image_id}."
            
            # assert rail[0][1] > rail[-1][1], f"Rails should be ordered from top to bottom (decreasing v), check image {image_id}."
            if rail[0][1] < rail[-1][1]:
                counter += 1
                print(f"Warning: Rail in image {image_id} is ordered from bottom to top (increasing v). This is not strictly required but we recommend ordering rails from top to bottom for consistency.")
                    
          
        for score in scores:
            assert isinstance(score, (int, float)), f"Each score should be a number, check image {image_id}."
            assert 0 <= score <= 1, f"Each confidence score should be between 0 and 1, but image {image_id} contains {score}."

    print("\nAll checks passed! Prediction file is correctly formatted.\n")