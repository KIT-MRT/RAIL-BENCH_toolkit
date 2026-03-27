"""
This file was developed with the assistance of AI-based tools (e.g., Claude). 
All content has been reviewed and adapted by the author, but AI-generated 
contributions may be present.
------------------------------------------------------------------------------

ChamferAP – Average Precision metric based on Chamfer Distance.

Follows the evaluation protocol used by MapTR / MapTRv2 for vectorised map
predictions:

1.  For every image, compute the full pair-wise Chamfer distance matrix
    between predicted and ground-truth polylines.
2.  Sort predictions by confidence (descending).
3.  Greedily assign each prediction to the closest *un-matched* GT polyline
    whose Chamfer distance is below a given threshold → **TP**, otherwise → **FP**.
4.  Accumulate TP/FP across all images (sorted globally by confidence),
    compute precision / recall, and derive AP (area under the PR curve).
5.  Report AP at each requested Chamfer-distance threshold and report
    the mean across thresholds (mAP).

Interface mirrors ``LineAP`` in ``metrics_v4.py`` so that it plugs into the
existing ``eval_tracks.py`` pipeline with no friction.
"""

from __future__ import annotations

import copy
import os
import numpy as np
from scipy.optimize import linear_sum_assignment
from typing import List, Dict, Optional, Tuple

from Benchmarks.RAILBENCH_Rail.metrics.ChamferAP.chamfer_distance import (
    sample_polyline,
    chamfer_distance_polylines,
)


class ChamferAP:
    """
    Compute Average Precision using Chamfer Distance for polyline matching,
    following the MapTR evaluation protocol.

    Parameters
    ----------
    predictions : dict
        ``{image_name: {'rails': [polyline, ...], 'score': [float, ...]}}``
        Each polyline is a list of ``[u, v]`` points.

    gt : dict
        Ground truth in RailBench / COCO format (same as ``LineAP``).

    num_sample_points : int
        Number of points to uniformly resample each polyline to before
        computing Chamfer distance (default 100).

    extended_summary : bool
        If True, per-image matching details are stored for later analysis /
        visualisation.
    """

    def __init__(
        self,
        predictions: dict,
        gt: dict,
        num_sample_points: int = 20,
        extended_summary: bool = False,
    ):
        self.predictions = self._process_predictions(predictions)
        self.gt = self._process_gt(gt)
        self._checks()

        self.num_sample_points = num_sample_points
        self.extended_summary = extended_summary
        self.results: Dict[str, dict] = {}

    # ------------------------------------------------------------------
    # Data preparation  (same helpers as LineAP)
    # ------------------------------------------------------------------

    @staticmethod
    def _process_gt(gt: dict) -> dict:
        """Convert RailBench/COCO GT format to ``{img_name: {'rails': …, 'ignore_areas': …}}``."""
        img_id_name = {img['id']: img['file_name'] for img in gt['images']}
        img_id_width = {img['id']: img['width'] for img in gt['images']}
        #file_names = list(img_id_name_mapping.values())
        gt_rails = {}
        for img_id, img_name in img_id_name.items():
            gt_rails[img_name] = {'rails': [], 'ignore_areas': [], 'image_width': img_id_width[img_id]}

        cat_id_name = {cat["id"]: cat["name"] for cat in gt["categories"]}

        for ann in gt["annotations"]:
            img_name = img_id_name[ann["image_id"]]
            cat_name = cat_id_name[ann["category_id"]]
            if cat_name == "rail":
                gt_rails[img_name]["rails"].append(ann["polyline"])
            elif cat_name == "ignore_area":
                gt_rails[img_name]["ignore_areas"].append(ann["polygon"])
        return gt_rails

    @staticmethod
    def _process_predictions(predictions: dict) -> dict:
        """Ensure polylines are oriented foreground → background (large v → small v).

        Returns a deep copy so the caller's data is never mutated.
        """
        predictions = copy.deepcopy(predictions)
        for _img, pred in predictions.items():
            for i, rail in enumerate(pred["rails"]):
                if len(rail) < 2:
                    continue
                if rail[0][1] < rail[-1][1]:  # start_v < end_v → flip
                    pred["rails"][i] = rail[::-1]
        return predictions

    def _checks(self):
        for img in self.gt:
            if img not in self.predictions:
                raise KeyError(
                    f"Image '{img}' present in GT but missing in predictions."
                )
        for img, pred in self.predictions.items():
            if not isinstance(pred, dict):
                raise TypeError(
                    f"Prediction for '{img}' must be a dict with 'rails' and 'score'."
                )
            if "rails" not in pred or "score" not in pred:
                raise KeyError(
                    f"Predictions for '{img}' must contain 'rails' and 'score' keys."
                )
            if len(pred["rails"]) != len(pred["score"]):
                raise ValueError(
                    f"#rails and #scores mismatch for '{img}': "
                    f"{len(pred['rails'])} vs {len(pred['score'])}."
                )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(
        self,
        chamfer_thresholds: Optional[List[float]] = None,
        use_abs_chamfer: bool = True,
    ) -> dict:
        """
        Run evaluation at each Chamfer-distance threshold.

        Parameters
        ----------
        chamfer_thresholds : list of float
            Chamfer-distance thresholds (in pixels). Depending on *use_abs_chamfer*, 
            the thresholds are interpreted as absolute pixel values or as relative fractions (in percentage) of image width. 
           
        use_abs_chamfer : bool
            If True, use absolute thresholds from *chamfer_thresholds*.
            If False, use relative thresholds from *rel_chamfer_thresholds*.

        Returns
        -------
        self.results : dict
            Keyed by ``'chamfer_thr_<t>'`` (absolute) or
            ``'rel_chamfer_thr_<t>'`` (relative) with
            ``{'AP': …, 'mean_chamfer': …}``.
        """
        self.results = {}

        abs_flag = use_abs_chamfer
        thr_list = chamfer_thresholds

        for thr in thr_list:
            print(f"Evaluating for Chamfer distance threshold = {thr} {'px' if abs_flag else '% (relative wrt image width)'} ...")
            key = f"chamfer_thr_{thr}" if abs_flag else f"rel_chamfer_thr_{thr}"
            self.results[key] = {}
            if self.extended_summary:
                self.results[key]["images"] = {}

            tp, fp, n_gt, all_chamfer, all_scores = self._compute_tp_fp(
                thr, abs_dist_flag=abs_flag, result_key=key
            )

            # Precision / Recall
            acc_tp = np.cumsum(tp)
            acc_fp = np.cumsum(fp)
            recall = acc_tp / max(n_gt, 1)
            precision = acc_tp / np.maximum(acc_tp + acc_fp, 1)

            ap, mpre, mrec, _ = self._calculate_ap_every_point(recall, precision)

            mean_chamfer = float(np.mean(all_chamfer)) if len(all_chamfer) > 0 else -1.0

            self.results[key]["AP"] = ap
            self.results[key]["mean_chamfer"] = mean_chamfer

        # Compute mAP across thresholds
        aps = [
            self.results[f"chamfer_thr_{t}" if abs_flag else f"rel_chamfer_thr_{t}"]["AP"]
            for t in thr_list
        ]
        self.results["mAP"] = float(np.mean(aps))

        return self.results

    def print_summary(self):
        """Print a compact summary table."""
        print("=" * 50)
        print("ChamferAP Results")
        print("=" * 50)
        for key, res in self.results.items():
            if key == "mAP":
                continue
            print(f"  {key}:")
            print(f"    AP              = {res['AP']:.4f}")
            print(f"    Mean Chamfer    = {res['mean_chamfer']:.4f}")
        if "mAP" in self.results:
            print("-" * 50)
            print(f"  mAP (across thresholds) = {self.results['mAP']:.4f}")
        print("=" * 50)

    def return_results(self) -> dict:
        return self.results

    # ------------------------------------------------------------------
    # Core evaluation logic
    # ------------------------------------------------------------------

    def _compute_tp_fp(
        self, threshold: float, abs_dist_flag: bool = True, result_key: str | None = None,
    ) -> Tuple[np.ndarray, np.ndarray, int, list, list]:
        """
        Accumulate TP / FP across all images for a single threshold.

        Parameters
        ----------
        threshold : float
            Distance threshold.  Interpreted as absolute pixels when
            *abs_dist_flag* is True, or as a fraction of image width when
            False.
        abs_dist_flag : bool
            Whether *threshold* is absolute (True) or relative (False).
        result_key : str or None
            Key into ``self.results`` used to store per-image extended
            summaries.

        Returns
        -------
        tp, fp : np.ndarray   – binary arrays (globally sorted by confidence)
        n_gt   : int           – total number of GT polylines
        all_chamfer : list     – Chamfer distances of matched (TP) pairs
        all_scores  : list     – corresponding confidence scores
        """
        # Collect per-image results first
        per_image: List[dict] = []
        n_gt_total = 0

        for img_name in self.gt:
            gt_rails = self.gt[img_name]["rails"].copy()
            pred_rails = self.predictions[img_name]["rails"].copy()
            pred_scores = list(self.predictions[img_name]["score"]).copy()
            n_gt_total += len(gt_rails)

            # Resolve per-image threshold
            if abs_dist_flag:
                img_threshold = threshold
            else:
                img_threshold = threshold/100 * self.gt[img_name]["image_width"]

            img_result = self._compute_tp_fp_single_image(
                pred_rails, pred_scores, gt_rails, img_threshold
            )
            per_image.append(img_result)

            if self.extended_summary and result_key is not None:
                self.results[result_key]["images"][img_name] = img_result

        # Flatten across images and sort globally by confidence (descending)
        all_tp: List[int] = []
        all_scores: List[float] = []
        all_chamfer: List[float] = []

        for r in per_image:
            all_tp.extend(r["tp_flags"])
            all_scores.extend(r["scores"])
            all_chamfer.extend(r["matched_chamfer"])

        all_tp_arr = np.array(all_tp, dtype=np.float64)
        all_scores_arr = np.array(all_scores, dtype=np.float64)

        # Sort by confidence (primary, descending) then by TP flag (secondary, descending – so TPs come first on ties)
        sort_idx = np.lexsort((all_tp_arr, all_scores_arr))[::-1]
        tp = all_tp_arr[sort_idx]
        fp = 1.0 - tp

        sorted_scores = [all_scores[i] for i in sort_idx]

        return tp, fp, n_gt_total, all_chamfer, sorted_scores

    def _compute_tp_fp_single_image(
        self,
        pred_rails: list,
        pred_scores: list,
        gt_rails: list,
        threshold: float,
    ) -> dict:
        """
        For one image, perform confidence-sorted greedy matching using
        Chamfer distance and return per-prediction TP/FP flags.

        This follows the MapTR evaluation protocol:
        - Sort predictions by confidence (descending).
        - Compute pairwise Chamfer distance matrix.
        - For each prediction (in confidence order), match to the closest
          unmatched GT if distance < threshold.

        Returns
        -------
        dict with keys:
            tp_flags        – list[int] of 0/1 per prediction (confidence-sorted)
            scores          – list[float] of confidence scores (same order)
            matched_chamfer – list[float] of Chamfer distances for TP matches
            cost_matrix     – np.ndarray (n_pred, n_gt) full Chamfer distance matrix
        """
        n_pred = len(pred_rails)
        n_gt = len(gt_rails)

        # Edge cases
        if n_pred == 0:
            return {
                "tp_flags": [],
                "scores": [],
                "matched_chamfer": [],
                "cost_matrix": np.empty((0, n_gt)),
            }
        if n_gt == 0:
            # All predictions are FP
            sorted_idx = np.argsort(pred_scores)[::-1]
            return {
                "tp_flags": [0] * n_pred,
                "scores": [pred_scores[i] for i in sorted_idx],
                "matched_chamfer": [],
                "cost_matrix": np.empty((n_pred, 0)),
            }

        # 1) Pairwise Chamfer distance matrix
        cost = np.zeros((n_pred, n_gt))
        for i, pl in enumerate(pred_rails):
            for j, gl in enumerate(gt_rails):
                cost[i, j] = chamfer_distance_polylines(
                    pl, gl, num_points=self.num_sample_points
                )

        # 2) Sort predictions by confidence (descending)
        sorted_idx = np.argsort(pred_scores)[::-1]

        tp_flags: List[int] = []
        scores: List[float] = []
        matched_chamfer: List[float] = []
        gt_matched = np.zeros(n_gt, dtype=bool)

        for idx in sorted_idx:
            scores.append(pred_scores[idx])
            # Find best unmatched GT
            dists = cost[idx].copy()
            dists[gt_matched] = np.inf
            best_gt = int(np.argmin(dists))
            best_dist = dists[best_gt]

            if best_dist < threshold:
                tp_flags.append(1)
                gt_matched[best_gt] = True
                matched_chamfer.append(float(best_dist))
            else:
                tp_flags.append(0)


        output = {
            "tp_flags": tp_flags,
            "scores": scores,
            "matched_chamfer": matched_chamfer,
            "cost_matrix": cost,
        }

        if self.extended_summary:
            output["pred_rails_sorted"] = [pred_rails[i] for i in sorted_idx]

        return output
    
    # ------------------------------------------------------------------
    # AP computation (same as LineAP / VOC-style all-point interpolation)
    # ------------------------------------------------------------------

    @staticmethod
    def _calculate_ap_every_point(
        rec: np.ndarray, prec: np.ndarray
    ) -> Tuple[float, list, list, list]:
        """
        All-point interpolated AP, identical to the implementation used in
        LineAP (originally from
        https://github.com/rafaelpadilla/review_object_detection_metrics).
        """
        mrec: list = [0.0] + list(rec) + [1.0]
        mpre: list = [0.0] + list(prec) + [0.0]

        # Make precision monotonically decreasing
        for i in range(len(mpre) - 1, 0, -1):
            mpre[i - 1] = max(mpre[i - 1], mpre[i])

        # Find points where recall changes
        ii = [i + 1 for i in range(len(mrec) - 1) if mrec[i + 1] != mrec[i]]

        ap = sum((mrec[i] - mrec[i - 1]) * mpre[i] for i in ii)

        return float(ap), mpre[: len(mpre) - 1], mrec[: len(mpre) - 1], ii
