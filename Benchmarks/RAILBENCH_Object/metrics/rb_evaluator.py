
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

from collections import defaultdict
import os

import numpy as np
from Benchmarks.RAILBENCH_Object.metrics.bounding_box import BBFormat, BBType


def get_coco_summary(groundtruth_bbs, detected_bbs, categories):
    """
    Parameters
        ----------
            groundtruth_bbs : list
                A list containing objects of type BoundingBox representing the ground-truth bounding boxes.
            detected_bbs : list
                A list containing objects of type BoundingBox representing the detected bounding boxes.
            categories : list
                A list of dictionaries, each containing 'id' and 'name' keys, representing the category mapping.
    Returns:
            A dictionary with one entry for each metric.
    """

    # category_mapping
    category_mapping = {c['id']: c['name'] for c in categories}

    # separate bbs per image X class
    _bbs = _group_detections(detected_bbs, groundtruth_bbs)

    # pairwise ious
    _ious = {k: _compute_ious(**v) for k, v in _bbs.items()}

    # pairwise iops
    _iops = {k: _comnpute_iops(**v) for k, v in _bbs.items()}

    max_dets_default=200

    def _evaluate(iou_threshold, max_dets, area_range, occlusion_range):
        # accumulate evaluations on a per-class basis
        _evals = defaultdict(lambda: {"scores": [], "matched": [], "NP": []})
        for img_id, class_id in _bbs:
            ev = _evaluate_image(
                _bbs[img_id, class_id]["dt"],
                _bbs[img_id, class_id]["gt"],
                _ious[img_id, class_id],
                _iops[img_id, class_id],
                iou_threshold,
                max_dets,
                area_range,
                occlusion_range,
                img_id=img_id,
                class_id=class_id,
            )
            acc = _evals[class_id]
            acc["scores"].append(ev["scores"])
            acc["matched"].append(ev["matched"])
            acc["NP"].append(ev["NP"])

        # now reduce accumulations
        for class_id in _evals:
            acc = _evals[class_id]
            acc["scores"] = np.concatenate(acc["scores"])
            acc["matched"] = np.concatenate(acc["matched"]).astype(bool)
            acc["NP"] = np.sum(acc["NP"])

        res = []
        # run ap calculation per-class
        for class_id in _evals:
            ev = _evals[class_id]
            res.append({
                "class": class_id,
                **_compute_ap_recall(ev["scores"], ev["matched"], ev["NP"]),
            })
        return res

    iou_thresholds = np.linspace(0.5, 0.95, int(np.round((0.95 - 0.5) / 0.05)) + 1, endpoint=True)

    # compute simple AP with all thresholds, using up to 200 dets, and all areas
    full = {
        i: _evaluate(iou_threshold=i, max_dets=max_dets_default, area_range=(0, np.inf), occlusion_range=None)
        for i in iou_thresholds
    }

    mAP50 = np.mean([x['AP'] for x in full[0.50] if x['AP'] is not None])
    mAP75 = np.mean([x['AP'] for x in full[0.75] if x['AP'] is not None])
    mAP = np.mean([x['AP'] for k in full for x in full[k] if x['AP'] is not None])

    AP_by_class = dict()
    for i, res in full.items():
        AP_by_class[i] = dict()
        for x in res:
            category_name = category_mapping[x['class']]
            AP_by_class[i][category_name] = x['AP']

    # max recall for 200 dets can also be calculated here
    mAR200 = np.mean(
        [x['TP'] / x['total positives'] for k in full for x in full[k] if x['TP'] is not None])
    
    #--------------------------------------------------------------------------------------------

    easy = {
        i: _evaluate(iou_threshold=i, max_dets=max_dets_default, area_range=(50**2, np.inf), occlusion_range=(0, 0))
        for i in iou_thresholds
    }
    mAPeasy = [x['AP'] for k in easy for x in easy[k] if x['AP'] is not None]
    mAPeasy = np.nan if mAPeasy == [] else np.mean(mAPeasy)
    mAReasy = [
        x['TP'] / x['total positives'] for k in easy for x in easy[k] if x['TP'] is not None
    ]
    mAReasy = np.nan if mAReasy == [] else np.mean(mAReasy)

    moderate = {
        i: _evaluate(iou_threshold=i, max_dets=max_dets_default, area_range=(30**2, np.inf), occlusion_range=(0, 2))
        for i in iou_thresholds
    }
    mAPmoderate = [x['AP'] for k in moderate for x in moderate[k] if x['AP'] is not None]
    mAPmoderate = np.nan if mAPmoderate == [] else np.mean(mAPmoderate)
    mARmoderate = [
        x['TP'] / x['total positives'] for k in moderate for x in moderate[k] if x['TP'] is not None
    ]
    mARmoderate = np.nan if mARmoderate == [] else np.mean(mARmoderate)

    hard = {
        i: _evaluate(iou_threshold=i, max_dets=max_dets_default, area_range=(0, np.inf), occlusion_range=(0, 4))
        for i in iou_thresholds
    }
    mAPhard = [x['AP'] for k in hard for x in hard[k] if x['AP'] is not None]
    mAPhard = np.nan if mAPhard == [] else np.mean(mAPhard)
    mARhard = [
        x['TP'] / x['total positives'] for k in hard for x in hard[k] if x['TP'] is not None
    ]
    mARhard = np.nan if mARhard == [] else np.mean(mARhard)
    mAPhard = mAP
    mARhard = mAR200


    #--------------------------------------------------------------------------------------------
    # restrict number of detections to smaller number than 200

    max_det1 = {
        i: _evaluate(iou_threshold=i, max_dets=1, area_range=(0, np.inf), occlusion_range=None)
        for i in iou_thresholds
    }
    mAR1 = np.mean([
        x['TP'] / x['total positives'] for k in max_det1 for x in max_det1[k] if x['TP'] is not None
    ])

    max_det10 = {
        i: _evaluate(iou_threshold=i, max_dets=10, area_range=(0, np.inf), occlusion_range=None)
        for i in iou_thresholds
    }
    mAR10 = np.mean([
        x['TP'] / x['total positives'] for k in max_det10 for x in max_det10[k]
        if x['TP'] is not None
    ])

    max_det100 = {
        i: _evaluate(iou_threshold=i, max_dets=100, area_range=(0, np.inf), occlusion_range=None)
        for i in iou_thresholds
    }
    mAR100 = np.mean([
        x['TP'] / x['total positives'] for k in max_det100 for x in max_det100[k]
        if x['TP'] is not None
    ])


    return {
        "AP": mAP,
        "AP50": mAP50,
        "AP75": mAP75,

        "APeasy": mAPeasy,
        "APmoderate": mAPmoderate,
        "APhard": mAPhard,
        "AReasy": mAReasy,
        "ARmoderate": mARmoderate,
        "ARhard": mARhard,

        "AP_by_class": AP_by_class,

        "AR1": mAR1,
        "AR10": mAR10,
        "AR100": mAR100,
        "AR200": mAR200,
    }


def get_coco_metrics(
        groundtruth_bbs,
        detected_bbs,
        iou_threshold=0.5,
        area_range=(0, np.inf),
        occlusion_level=None,
        max_dets=100,
):
    """ Calculate the Average Precision and Recall metrics as in COCO's official implementation
        given an IOU threshold, area range and maximum number of detections.
    Parameters
        ----------
            groundtruth_bbs : list
                A list containing objects of type BoundingBox representing the ground-truth bounding boxes.
            detected_bbs : list
                A list containing objects of type BoundingBox representing the detected bounding boxes.
            iou_threshold : float
                Intersection Over Union (IOU) value used to consider a TP detection.
            area_range : (numerical x numerical)
                Lower and upper bounds on annotation areas that should be considered.
            occlusion_level : int
                The occlusion level of the ground truth bounding boxes to be considered. 
            max_dets : int
                Upper bound on the number of detections to be considered for each class in an image.

    Returns:
            A list of dictionaries. One dictionary for each class.
            The keys of each dictionary are:
            dict['class']: class representing the current dictionary;
            dict['precision']: array with the precision values;
            dict['recall']: array with the recall values;
            dict['AP']: average precision;
            dict['interpolated precision']: interpolated precision values;
            dict['interpolated recall']: interpolated recall values;
            dict['total positives']: total number of ground truth positives;
            dict['TP']: total number of True Positive detections;
            dict['FP']: total number of False Positive detections;

            if there was no valid ground truth for a specific class (total positives == 0),
            all the associated keys default to None
    """

    # separate bbs per image X class
    _bbs = _group_detections(detected_bbs, groundtruth_bbs)

    # pairwise ious
    _ious = {k: _compute_ious(**v) for k, v in _bbs.items()}

    # pairwise iops
    _iops = {k: _comnpute_iops(**v) for k, v in _bbs.items()}

    # accumulate evaluations on a per-class basis
    _evals = defaultdict(lambda: {"scores": [], "matched": [], "NP": []})

    for img_id, class_id in _bbs:
        ev = _evaluate_image(
            _bbs[img_id, class_id]["dt"],
            _bbs[img_id, class_id]["gt"],
            _ious[img_id, class_id],
            _iops[img_id, class_id],
            iou_threshold,
            max_dets,
            area_range,
            occlusion_level,
            img_id=img_id,
            class_id=class_id,
        )
        acc = _evals[class_id]
        acc["scores"].append(ev["scores"])
        acc["matched"].append(ev["matched"])
        acc["NP"].append(ev["NP"])

    # now reduce accumulations
    for class_id in _evals:
        acc = _evals[class_id]
        acc["scores"] = np.concatenate(acc["scores"])
        acc["matched"] = np.concatenate(acc["matched"]).astype(bool)
        acc["NP"] = np.sum(acc["NP"])

    res = {}
    # run ap calculation per-class
    for class_id in _evals:
        ev = _evals[class_id]
        res[class_id] = {
            "class": class_id,
            **_compute_ap_recall(ev["scores"], ev["matched"], ev["NP"])
        }
    return res


def _group_detections(dt, gt):
    """ simply group gts and dts on a imageXclass basis """
    bb_info = defaultdict(lambda: {"dt": [], "gt": []})
    for d in dt:
        i_id = d.get_image_name()
        c_id = d.get_class_id()
        bb_info[i_id, c_id]["dt"].append(d)
    for g in gt:
        i_id = g.get_image_name()
        c_id = g.get_class_id()
        bb_info[i_id, c_id]["gt"].append(g)
    return bb_info


def _get_area(a):
    """ COCO does not consider the outer edge as included in the bbox """
    x, y, x2, y2 = a.get_absolute_bounding_box(format=BBFormat.XYX2Y2)
    return (x2 - x) * (y2 - y)


def _compute_areas(a, b):
    """
    compute areas of a, b and intersection
    """
    xa, ya, x2a, y2a = a.get_absolute_bounding_box(format=BBFormat.XYX2Y2)
    xb, yb, x2b, y2b = b.get_absolute_bounding_box(format=BBFormat.XYX2Y2)

    # innermost left x
    xi = max(xa, xb)
    # innermost right x
    x2i = min(x2a, x2b)
    # same for y
    yi = max(ya, yb)
    y2i = min(y2a, y2b)

    # calculate areas
    Aa = max(x2a - xa, 0) * max(y2a - ya, 0) # area of bounding box a
    Ab = max(x2b - xb, 0) * max(y2b - yb, 0) # area of bounding box b
    Ai = max(x2i - xi, 0) * max(y2i - yi, 0) # area of intersection

    return Aa, Ab, Ai

def _compute_ious(dt, gt):
    """ compute pairwise ious """

    ious = np.zeros((len(dt), len(gt)))
    for g_idx, g in enumerate(gt):
        for d_idx, d in enumerate(dt):
            ad, ag, ai = _compute_areas(d, g)
            ious[d_idx, g_idx] = ai / (ad + ag - ai)
    return ious

def _comnpute_iops(dt, gt):
    """ compute pairwise iops """

    iops = np.zeros((len(dt), len(gt)))
    for g_idx, g in enumerate(gt):
        for d_idx, d in enumerate(dt):
            ad, _, ai = _compute_areas(d, g)
            if ad == 0:
                iops[d_idx, g_idx] = 0
            else:
                iops[d_idx, g_idx] = ai / ad  
    return iops


def _evaluate_image(dt, gt, ious, iops, iou_threshold, max_dets=None, area_range=None, occlusion_range=None, img_id=None, class_id=None):
    """ 
    Associate detections to ground truths using a mixture of COCO's method and Open Image Dataset V7.
    
    returns a dictionary with the following keys:
        - 'scores': list of predicted confidence values for each non-ignored detection
        - 'matched_dt': list of booleans indicating if a non-ignored detection was matched
        - 'matched_gt': list of booleans indicating if a non-ignored ground truth was matched
        - 'NP': number of non-ignored ground truths
    Note: a gt is ignored if its area is outside the specified range or not within a specific occlusion range; 
          a prediction that is matched to an ignored gt is also ignored or if it is not matched a outside the specified area range.
    """

    # sort dts by confidence (first: highest confidence)
    dt_sort = np.argsort([-d.get_confidence() for d in dt], kind="stable")

    # sort list of dts and chop by max dets
    dt = [dt[idx] for idx in dt_sort[:max_dets]]
    ious = ious[dt_sort[:max_dets]]
    iops = iops[dt_sort[:max_dets]]

    # generate ignored gt list by area_range (Note: also crowd gt can be ignored)
    def _is_ignore(bb):
        if bb.is_ignore():
            return True
        
        if area_range is None: 
            ignore_by_area = False
        else:
            ignore_by_area = not (area_range[0] <= _get_area(bb) <= area_range[1])

        if occlusion_range is None or bb.get_bb_type() == BBType.DETECTED:
            ignore_by_occlusion = False
        else:
            ignore_by_occlusion = not (occlusion_range[0] <= bb.get_occlusion() <= occlusion_range[1])

        return ignore_by_area or ignore_by_occlusion

    gt_ignore = [_is_ignore(g) for g in gt]
    gt_iscrowd = [g.iscrowd() for g in gt]

    # creating sorting categories for gt
    sort_cat = np.ones(len(gt), dtype=np.int8)
    sort_cat[gt_iscrowd] = 2
    sort_cat[gt_ignore] = 3

    # sort gts in order of regular, crowd, and ignore
    gt_sort = np.argsort(sort_cat, kind="stable")
    gt = [gt[idx] for idx in gt_sort]

    gt_reg = [not(gt_ignore[idx] or gt_iscrowd[idx]) for idx in gt_sort]
    gt_ignore = [gt_ignore[idx] for idx in gt_sort]
    gt_iscrowd = [gt_iscrowd[idx] for idx in gt_sort]

    ious = ious[:, gt_sort]
    iops = iops[:, gt_sort]

    # gtm = defaultdict(list) # each gt can be matched to multiple dts (if crowd)
    gtm = {}
    dtm = {} # each dt can be matched to only one gt
    ignore_dt_crowd = [] # list of dt indices that are matched to crowd gt

    for d_idx, d in enumerate(dt):
        # information about best match so far (m=-1 -> unmatched)
        iou = min(iou_threshold, 1 - 1e-10)
        iop = min(iou_threshold, 1 - 1e-10)
        m = -1
        for g_idx, g in enumerate(gt):
            # if this gt already matched, and not a crowd, continue
            if g_idx in gtm and not g.iscrowd():
                continue

            # if dt matched to regular gt, and on crowd gt or ignore gt, stop
            # if dt matched to a non-ignored crowd gt, and on ignore gt, stop
            if m > -1 and ((gt_reg[m] == True and gt_reg[g_idx] == False) or (gt_iscrowd[m] == True and (not gt_ignore[m]) and gt_ignore[g_idx] == True)):
                break

            # if current gt is regular, compar ious
            # if current gt is crowd, compare iops
            if (not gt_iscrowd[g_idx] and ious[d_idx, g_idx] < iou) or (gt_iscrowd[g_idx] and iops[d_idx, g_idx] < iop):
                continue

            if gt_iscrowd[g_idx] == True:
                iop = iops[d_idx, g_idx]
            else:
                iou = ious[d_idx, g_idx]
            m = g_idx

        # if match made store id of match for both dt and gt
        if m == -1:
            continue

        # for matches to a crowd gt, we only store the dt with highest confidence prediction, the other dts are ignored 
        if m in gtm:
            if gt_iscrowd[m] == True:
                ignore_dt_crowd.append(d_idx)
            else:
                print("Warning: dt matched to non-crowd gt that was already matched to a crowd gt")
        else:
            dtm[d_idx] = m
            gtm[m] = d_idx
            # gtm[m].append(d_idx)


    # generate ignore list for dts
    dt_ignore = []
    for d_idx, d in enumerate(dt):
        if d_idx in dtm:
            dt_ignore.append(gt_ignore[dtm[d_idx]])
        elif d_idx in ignore_dt_crowd:
            dt_ignore.append(True)
        else:
            dt_ignore.append(_is_ignore(d))

    # dt_ignore = [
    #     gt_ignore[dtm[d_idx]] if d_idx in dtm else _is_ignore(d) for d_idx, d in enumerate(dt)
    # ]

    # get score (confidence value) and "is matched?" for non-ignored dts
    scores = [dt[d_idx].get_confidence() for d_idx in range(len(dt)) if not dt_ignore[d_idx]]
    matched = [d_idx in dtm for d_idx in range(len(dt)) if not dt_ignore[d_idx]]

    n_gts = len([g_idx for g_idx in range(len(gt)) if not gt_ignore[g_idx]])

    # if os.environ.get("DEBUG_MODE") == "1" or os.environ.get("EVAL_CREATE_FIGURE") == "1":
    #     matched_dt_idx = list(dtm.keys())
    #     matched_gt_idx = list(gtm.keys())

    #     vis_matching(dt=dt, gt=gt, 
    #                  matched_dt_idx=matched_dt_idx, 
    #                  matched_gt_idx=matched_gt_idx, 
    #                  dt_ignore=dt_ignore, 
    #                  gt_ignore=gt_ignore, 
    #                  iou_threshold=iou_threshold, 
    #                  max_dets=max_dets, 
    #                  area_range=area_range, 
    #                  occlusion_level=occlusion_level,
    #                  img_id=img_id,
    #                  class_id=class_id)

    return {"scores": scores, "matched": matched, "NP": n_gts}


def _compute_ap_recall(scores, matched, NP, recall_thresholds=None):
    """ This curve tracing method has some quirks that do not appear when only unique confidence thresholds
    are used (i.e. Scikit-learn's implementation), however, in order to be consistent, the COCO's method is reproduced. """
    if NP == 0:
        return {
            "precision": None,
            "recall": None,
            "AP": None,
            "interpolated precision": None,
            "interpolated recall": None,
            "total positives": None,
            "TP": None,
            "FP": None
        }

    # by default evaluate on 101 recall levels
    if recall_thresholds is None:
        recall_thresholds = np.linspace(0.0,
                                        1.00,
                                        int(np.round((1.00 - 0.0) / 0.01)) + 1,
                                        endpoint=True)

    # sort in descending score order
    inds = np.argsort(-scores, kind="stable")

    scores = scores[inds]
    matched = matched[inds]

    tp = np.cumsum(matched)
    fp = np.cumsum(~matched)

    rc = tp / NP
    pr = tp / (tp + fp)

    # make precision monotonically decreasing
    i_pr = np.maximum.accumulate(pr[::-1])[::-1]

    rec_idx = np.searchsorted(rc, recall_thresholds, side="left")
    n_recalls = len(recall_thresholds)

    # get interpolated precision values at the evaluation thresholds
    i_pr = np.array([i_pr[r] if r < len(i_pr) else 0 for r in rec_idx])

    return {
        "precision": pr,
        "recall": rc,
        "AP": np.mean(i_pr),
        "interpolated precision": i_pr,
        "interpolated recall": recall_thresholds,
        "total positives": NP,
        "TP": tp[-1] if len(tp) != 0 else 0,
        "FP": fp[-1] if len(fp) != 0 else 0
    }
