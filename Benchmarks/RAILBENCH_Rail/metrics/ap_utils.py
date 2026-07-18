"""Shared all-point interpolated AP computation, used by both LineAP and
ChamferAP. Originally from
https://github.com/rafaelpadilla/review_object_detection_metrics.
"""
from typing import List, Tuple

import numpy as np


def calculate_ap_every_point(
    rec: np.ndarray, prec: np.ndarray
) -> Tuple[float, List[float], List[float], List[int]]:
    mrec: list = [0.0] + list(rec) + [1.0]
    mpre: list = [0.0] + list(prec) + [0.0]

    # Make precision monotonically decreasing
    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])

    # Find points where recall changes
    ii = [i + 1 for i in range(len(mrec) - 1) if mrec[i + 1] != mrec[i]]

    ap = sum((mrec[i] - mrec[i - 1]) * mpre[i] for i in ii)

    return float(ap), mpre[: len(mpre) - 1], mrec[: len(mpre) - 1], ii
