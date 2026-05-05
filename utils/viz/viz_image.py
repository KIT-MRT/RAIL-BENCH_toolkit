import cv2

def image_preparation(img_path, brightness_increase=0):
    """
    Load image and convert to RGB. Optionally increase brightness.

    Args:
    img_path: path to the image
    brightness_increase: value to increase brightness (default 0, no change)
    """
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if brightness_increase > 0:
        alpha = 1.0  # contrast control (1.0 = no change)
        beta = brightness_increase    # brightness control (increase to make brighter)
        img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

    return img