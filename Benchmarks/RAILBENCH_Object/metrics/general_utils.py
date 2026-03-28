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

def get_classes_from_txt_file(filepath_classes_det):
    classes = {}
    f = open(filepath_classes_det, 'r')
    id_class = 0
    for id_class, line in enumerate(f.readlines()):
        classes[id_class] = line.replace('\n', '')
    f.close()
    return classes


def replace_id_with_classes(bounding_boxes, filepath_classes_det):
    classes = get_classes_from_txt_file(filepath_classes_det)
    for bb in bounding_boxes:
        if not is_str_int(bb.get_class_id()):
            print(
                f'Warning: Class id represented in the {filepath_classes_det} is not a valid integer.'
            )
            return bounding_boxes
        class_id = int(bb.get_class_id())
        if class_id not in range(len(classes)):
            print(
                f'Warning: Class id {class_id} is not in the range of classes specified in the file {filepath_classes_det}.'
            )
            return bounding_boxes
        bb._class_id = classes[class_id]
    return bounding_boxes


def convert_box_xywh2xyxy(box):
    arr = box.copy()
    arr[:, 2] += arr[:, 0]
    arr[:, 3] += arr[:, 1]
    return arr


def convert_box_xyxy2xywh(box):
    arr = box.copy()
    arr[:, 2] -= arr[:, 0]
    arr[:, 3] -= arr[:, 1]
    return arr


# size => (width, height) of the image
# box => (X1, X2, Y1, Y2) of the bounding box
def convert_to_relative_values(size, box):
    dw = 1. / (size[0])
    dh = 1. / (size[1])
    cx = (box[1] + box[0]) / 2.0
    cy = (box[3] + box[2]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = cx * dw
    y = cy * dh
    w = w * dw
    h = h * dh
    # YOLO's format
    # x,y => (bounding_box_center)/width_of_the_image
    # w => bounding_box_width / width_of_the_image
    # h => bounding_box_height / height_of_the_image
    return (x, y, w, h)


# size => (width, height) of the image
# box => (centerX, centerY, w, h) of the bounding box relative to the image
def convert_to_absolute_values(size, box):
    w_box = size[0] * box[2]
    h_box = size[1] * box[3]

    x1 = (float(box[0]) * float(size[0])) - (w_box / 2)
    y1 = (float(box[1]) * float(size[1])) - (h_box / 2)
    x2 = x1 + w_box
    y2 = y1 + h_box
    return (round(x1), round(y1), round(x2), round(y2))


def is_str_int(s):
    if s[0] in ('-', '+'):
        return s[1:].isdigit()
    return s.isdigit()

