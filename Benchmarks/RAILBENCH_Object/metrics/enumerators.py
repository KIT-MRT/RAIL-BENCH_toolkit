"""
Attribution
-----------
This file is adapted from the `review_object_detection_metrics` project by Rafael Padilla
and contributors.

Original source:
    Rafael Padilla (and contributors), "review_object_detection_metrics"
    https://github.com/rafaelpadilla/review_object_detection_metrics

Original copyright notice (MIT License):
    Copyright (c) 2020 Rafael Padilla
    Permission is hereby granted, free of charge, to any person obtaining a copy of this
    software and associated documentation files (the "Software"), to deal in the Software
    without restriction, including without limitation the rights to use, copy, modify, merge,
    publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons
    to whom the Software is furnished to do so, subject to the following conditions:
    The above copyright notice and this permission notice shall be included in all copies or
    substantial portions of the Software.

Adaptation
----------
This code has been adapted and integrated into the RAIL-BENCH Toolkit
and may contain modifications for project-specific behaviour.

Date: 2026-02-17

Citation
--------
When publishing results produced with this code, please cite the original work:
    Padilla R, Passos WL, Dias TLB, Netto SL, da Silva EAB.
    A Comparative Analysis of Object Detection Metrics with a Companion Open-Source Toolkit.
    Electronics. 2021; 10(3):279. https://doi.org/10.3390/electronics10030279
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

