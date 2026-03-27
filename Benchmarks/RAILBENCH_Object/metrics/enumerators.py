"""
Attribution
-----------
This file is adapted from the `review_object_detection_metrics` project by Rafael Padilla
and contributors.

Original source:
    Rafael Padilla (and contributors), "review_object_detection_metrics"
    https://github.com/rafaelpadilla/review_object_detection_metrics

Adaptation
----------
This code has been adapted and integrated into the RailBench Benchmark Suite
and may contain modifications for project-specific behaviour.

Date: 2026-02-17

License & citation
------------------
Please consult the original repository for the authoritative license and citation
information. When using or redistributing this code, comply with the original
project's license and cite the original work where appropriate.
"""

from enum import Enum

class CoordinatesType(Enum):
    """
    Class representing if the coordinates are relative to the
    image size or are absolute values.

        Developed by: Rafael Padilla
        Last modification: Apr 28 2018
    """
    RELATIVE = 1
    ABSOLUTE = 2


class BBType(Enum):
    """
    Class representing if the bounding box is groundtruth or not.
    """
    GROUND_TRUTH = 1
    DETECTED = 2


class BBFormat(Enum):
    """
    Class representing the format of a bounding box.
    """
    XYWH = 1
    XYX2Y2 = 2
    PASCAL_XML = 3
    YOLO = 4

