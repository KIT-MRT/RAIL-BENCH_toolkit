from utils.helpers import load_json
import argparse
import os
from PIL import Image
import numpy as np

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_path", type=str, help="Path to prediction folder")

    args = parser.parse_args()

    print("\nChecking predictions for RAIL-BENCH Vegetation Challenge...")
    print("Check masks in folder:", args.pred_path)

    file_names = sorted([f for f in os.listdir(args.pred_path) if f.endswith('.png')])
    challenge_images_info = load_json('format/railbench_test_image_infos.json')
    test_images = list(challenge_images_info.keys())

    # Check number of masks and file names
    assert len(file_names) == len(test_images), f"Number of predicted masks ({len(file_names)}) does not match number of required masks ({len(test_images)})."

    for file_name in file_names:
        assert file_name in test_images, f"Predicted mask {file_name} not found in required masks."

    # Check masks 
    for file_name in file_names:
        mask_path = os.path.join(args.pred_path, file_name)
        mask = Image.open(mask_path)
        mask_array = np.array(mask)

        # Check dimensions
        required_shape = challenge_images_info[file_name]['shape']
        mask_shape = list(mask_array.shape)
        assert mask_shape == required_shape, f"Mask {file_name} has incorrect dimensions {mask_shape}, expected {required_shape}."

        # Check data type and unique values
        unique_values = np.unique(mask_array)
        assert set(unique_values).issubset({0, 1, 2}), f"Mask {file_name} contains invalid pixel values {unique_values}, expected only 0, 1, and 2."
    
        assert mask_array.dtype == np.uint8, f"Mask {file_name} has incorrect data type {mask_array.dtype}, expected uint8."

    print("\nAll checks passed! Predicted masks are correctly named and formatted.\n")

