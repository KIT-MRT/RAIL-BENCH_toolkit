import cv2

def image_preparation(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # make image brighter
    # alpha = 1.0  # contrast control (1.0 = no change)
    # beta = 50    # brightness control (increase to make brighter)
    # img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

    return img