"""
This file was developed with the assistance of AI-based tools (e.g., Claude). 
All content has been reviewed and adapted by the author, but AI-generated 
contributions may be present.
------------------------------------------------------------------------------

ChamferAP stands for Average Precision metric based on Chamfer Distance.

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
"""

from __future__ import annotations

import copy
import numpy as np
from typing import List, Dict, Optional, Tuple

from Benchmarks.RAILBENCH_Rail.metrics.ChamferAP.chamfer_distance import chamfer_distance_polylines
from Benchmarks.RAILBENCH_Rail.utils.ap_utils import calculate_ap_every_point
from Benchmarks.RAILBENCH_Rail.utils.track_width import track_width_line_parameters
from Benchmarks.RAILBENCH_Rail.utils.polyline_tools import polyline_orientation
from Benchmarks.RAILBENCH_Rail.utils.ignore_areas import remove_preds_in_ignore

# default line parameters for track width calculation if no valid line can be fitted
M_DEFAULT = 0.52
B_DEFAULT = -520

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
        computing Chamfer distance (default 50).

    dist_thr_mode : str
        Mode for determining the distance threshold. Default: 'track_width'.
         Options: 
            1. 'absolute': distance threshold is interpreted as an absolute distance in pixels.
            2. 'relative': distance threshold is interpreted as a relative distance in percentage with respect to the image width.
            3. 'track_width': distance threshold is interpreted as a relative distance in percentage with respect to the track width.

    extended_summary : bool
        If True, per-image matching details are stored for later analysis /
        visualisation.

    max_detections : int, optional
        If set, only the top ``max_detections`` highest-confidence predicted rails per image are considered
        for matching (analogous to COCO's ``maxDets``). This penalizes over-generation of low-confidence
        predictions: precision is computed only from the kept predictions, while recall is still measured
        against the full, uncapped set of GT rails. If None, no cap is applied. Default 100.

    verbose: bool
        If True, print progress messages during evaluation. Default: False.
    """

    def __init__(
        self,
        predictions: dict,
        gt: dict,
        num_sample_points: int = 50,
        dist_thr_mode='track_width',
        extended_summary: bool = False,
        max_detections: Optional[int] = 100,
        verbose: bool = False
    ):
        self.dist_thr_mode = dist_thr_mode

        predictions_filtered = remove_preds_in_ignore(predictions, gt)
        self.predictions = self._process_predictions(predictions_filtered)
        self.gt = self._process_gt(gt, self.dist_thr_mode)
        self._checks()

        self.num_sample_points = num_sample_points
        self.extended_summary = extended_summary

        assert max_detections is None or max_detections > 0, "max_detections must be None or a positive integer."
        self.max_detections = max_detections

        self.results: Dict[str, dict] = {}

        self.verbose = verbose


    # ------------------------------------------------------------------
    # Data preparation  (same helpers as LineAP)
    # ------------------------------------------------------------------

    @staticmethod
    def _process_gt(gt: dict, dist_thr_mode: str) -> dict:
        """Convert RailBench/COCO GT format to ``{img_name: {'rails': …, 'ignore_areas': …}}``."""
        img_id_name = {img['id']: img['file_name'] for img in gt['images']}
        img_id_width = {img['id']: img['width'] for img in gt['images']}

        gt_rails = {}
        for img_id, img_name in img_id_name.items():
            gt_rails[img_name] = {'rails': [], 'ignore_areas': [], 'image_width': img_id_width[img_id]}
            if dist_thr_mode == 'track_width':
                gt_rails[img_name]['track_ids'] = []
                gt_rails[img_name]['rightRail'] = []
                gt_rails[img_name]['track_width_line_parameters'] = {'m': None, 'b': None}

        cat_id_name = {cat["id"]: cat["name"] for cat in gt["categories"]}

        for ann in gt["annotations"]:
            img_name = img_id_name[ann["image_id"]]
            cat_name = cat_id_name[ann["category_id"]]
            if cat_name == "rail":
                gt_rails[img_name]["rails"].append(polyline_orientation(ann["polyline"]))
                if dist_thr_mode == 'track_width':
                        gt_rails[img_name]['track_ids'].append(ann['track_id'])
                        gt_rails[img_name]['rightRail'].append(ann['rightRail'])
            elif cat_name == "ignore_area":
                gt_rails[img_name]["ignore_areas"].append(ann["polygon"])

        # Compute line parameters for track width calculation if dist_thr_mode is 'track_width'
        if dist_thr_mode == 'track_width':
            for img_name, img_data in gt_rails.items():
                m, b = track_width_line_parameters(gt_rails = img_data['rails'],
                                                   track_ids = img_data['track_ids'],
                                                   right_rail = img_data['rightRail'])
                if m is None or b is None:
                    m = M_DEFAULT
                    b = B_DEFAULT
                gt_rails[img_name]['track_width_line_parameters']['m'] = m
                gt_rails[img_name]['track_width_line_parameters']['b'] = b
        
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
                rail = polyline_orientation(rail)
                pred["rails"][i] = rail 
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

    def _cap_predictions(self, pred_rails: list, pred_scores: list) -> Tuple[list, list]:
        """
        Keep only the top-`max_detections` highest-confidence predicted rails
        (analogous to COCO's maxDets), to penalize over-generation. No-op if
        max_detections is None or there are already fewer predictions than the cap.
        """
        if self.max_detections is None or len(pred_rails) <= self.max_detections:
            return pred_rails, pred_scores
        sorted_idx = np.argsort(pred_scores)[::-1][:self.max_detections]
        return [pred_rails[i] for i in sorted_idx], [pred_scores[i] for i in sorted_idx]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(
        self,
        dist_thresholds: Optional[List[float]] = None,
        min_dist_threshold: Optional[float] = None,
    ) -> dict:
        """
        Run evaluation at each Chamfer-distance threshold.

        Parameters
        ----------
        dist_thresholds : list of float
            Distance thresholds. 
            Depending on dist_thr_mode, these are interpreted as absolute pixel values (if 'absolute'), relative fractions of image width (if 'relative'), or relative fractions of track width (if 'track_width').

        min_dist_threshold : float, optional
            Minimum distance threshold to consider for evaluation with respect to track width. If provided, any threshold below this value will be set to this minimum value. 

        Returns
        -------
        self.results : dict
            Keyed by ``'chamfer_thr_<t>'`` (absolute) or
            ``'rel_chamfer_thr_<t>'`` (relative) with
            ``{'AP': …, 'mean_chamfer': …}``.
        """
        self.results = {}
        self._cost_matrix_cache: Dict[str, np.ndarray] = {}

        thr_list = dist_thresholds

        if self.dist_thr_mode == 'track_width':
            assert min_dist_threshold is not None, "min_dist_threshold must be provided when dist_thr_mode is 'track_width'."

        for thr in thr_list:
            if self.dist_thr_mode == 'absolute':
                if self.verbose:
                    print(f"Evaluating for Chamfer distance threshold = {thr} px ...")
                key = f"chamfer_thr_{thr}"
            elif self.dist_thr_mode == 'relative':
                if self.verbose:
                    print(f"Evaluating for Chamfer distance threshold = {thr}% (relative wrt image width) ...")
                key = f"rel_chamfer_thr_{thr}"
            elif self.dist_thr_mode == 'track_width':
                if self.verbose:
                    print(f"Evaluating for Chamfer distance threshold = {thr}% (relative wrt track width) ...")
                key = f"rel_chamfer_thr_{thr}"
            else:
                raise ValueError(f"Invalid dist_thr_mode: {self.dist_thr_mode}")
            
            self.results[key] = {}
            if self.extended_summary:
                self.results[key]["images"] = {}

            tp, fp, n_gt, all_chamfer, all_scores = self._compute_tp_fp(
                thr, result_key=key, min_dist_threshold=min_dist_threshold
            )

            # Precision / Recall
            acc_tp = np.cumsum(tp)
            acc_fp = np.cumsum(fp)
            recall = acc_tp / max(n_gt, 1)
            precision = acc_tp / np.maximum(acc_tp + acc_fp, 1)

            ap, mpre, mrec, _ = calculate_ap_every_point(recall, precision)

            mean_chamfer = float(np.mean(all_chamfer)) if len(all_chamfer) > 0 else -1.0

            self.results[key]["AP"] = ap
            self.results[key]["mean_chamfer"] = mean_chamfer

        # Compute mAP across thresholds
        aps = [
            self.results[f"chamfer_thr_{t}" if self.dist_thr_mode == 'absolute' else f"rel_chamfer_thr_{t}"]["AP"]
            for t in thr_list
        ]
        self.results["mAP"] = float(np.mean(aps))

        return self.results

    def _format_threshold_label(self, key: str) -> str:
        """Human-readable label for a per-threshold results key, based on dist_thr_mode."""
        d_t = float(key.split("_")[-1])
        if self.dist_thr_mode == 'absolute':
            return f"Distance threshold = {d_t:g} px"
        elif self.dist_thr_mode == 'relative':
            return f"Distance threshold = {d_t:g}% (relative to image width)"
        elif self.dist_thr_mode == 'track_width':
            return f"Distance threshold = {d_t:g}% (relative to track width)"
        else:
            raise ValueError(f"Invalid dist_thr_mode: {self.dist_thr_mode}")

    def print_summary(self):
        """Print a compact summary table."""
        print("=" * 50)
        print("ChamferAP Results")
        print("=" * 50)
        for key, res in self.results.items():
            if key == "mAP":
                continue
            print(f"  {self._format_threshold_label(key)}:")
            print(f"    AP                    = {res['AP']:.4f}")
            print(f"    Mean Chamfer dist.    = {res['mean_chamfer']:.4f}")
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
        self, 
        threshold: float, 
        result_key: str | None = None,
        min_dist_threshold: Optional[float] = None
    ) -> Tuple[np.ndarray, np.ndarray, int, list, list]:
        """
        Accumulate TP / FP across all images for a single threshold.

        Parameters
        ----------
        threshold : float
            Distance threshold. 
        result_key : str or None
            Key into ``self.results`` used to store per-image extended
            summaries.
        min_dist_threshold : float, optional
            Minimum distance threshold to consider for evaluation with respect to track width. If provided, any threshold below this value will be set to this minimum value.

        Returns
        -------
        - tp, fp : np.ndarray - binary arrays (globally sorted by confidence)
        - n_gt   : int - total number of GT polylines
        - all_chamfer : list - Chamfer distances of matched (TP) pairs
        - all_scores  : list - corresponding confidence scores
        """
        # Collect per-image results first
        per_image: List[dict] = []
        n_gt_total = 0

        for img_name in self.gt:
            gt_rails = self.gt[img_name]["rails"].copy()
            pred_rails = self.predictions[img_name]["rails"].copy()
            pred_scores = list(self.predictions[img_name]["score"]).copy()
            pred_rails, pred_scores = self._cap_predictions(pred_rails, pred_scores)
            track_width_line_params = self.gt[img_name].get('track_width_line_parameters', None)
            n_gt_total += len(gt_rails)

            # Resolve per-image threshold
            img_threshold = self._resolve_dist_thres(img_name, threshold)
            if self.dist_thr_mode == 'track_width':
                img_threshold_list = self._compute_rail_dist_threshold(gt_rails, track_width_line_params, img_threshold, min_dist_threshold=min_dist_threshold)
            else:
                img_threshold_list = None

            if img_name not in self._cost_matrix_cache:
                self._cost_matrix_cache[img_name] = self._compute_cost_matrix_single_image(
                    pred_rails, gt_rails
                )
            cost = self._cost_matrix_cache[img_name]

            img_result = self._match_single_image(
                cost, pred_rails, pred_scores, threshold=img_threshold, threshold_list=img_threshold_list
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

    def _resolve_dist_thres(self, img_ident, dist_thres):
        if self.dist_thr_mode == 'absolute':
            return dist_thres
        elif self.dist_thr_mode == 'relative':
            return int(dist_thres/100.0 * self.gt[img_ident]['image_width'])
        else:  # 'track_width'
            # Transform percentage into scale factor
            return dist_thres/100.0

    def _compute_rail_dist_threshold(self, gt_rails: list, track_width_line_params: dict, scale: float, min_dist_threshold: float) -> float:
        """
        Assign a distance threshold to each individual rail based on its track width, using the provided line parameters.

        Parameters:
        gt_rails : list
            List of ground truth rails (polylines). 
        track_width_line_params : dict
            Dictionary containing the line parameters 'm' and 'b' for the track width calculation.
        scale : float
            The scale factor to be applied to the track width.
        min_dist_threshold : float
            Minimum distance threshold to consider for evaluation with respect to track width. If provided, any threshold below this value will be set to this minimum value.

        Returns:
        img_threshold_list : list
            List of distance thresholds for each rail, computed as scale * track_width, with a minimum of min_dist_threshold if provided.
        """

        m = track_width_line_params.get('m', M_DEFAULT)
        b = track_width_line_params.get('b', B_DEFAULT)

        # Compute the average track width for the given rails
        track_widths = []
        for rail in gt_rails:
            if len(rail) < 2:
                raise ValueError("Each rail must have at least two points to compute track width.")
            start_v = rail[0][1]
            end_v = rail[-1][1]
            avg_v = (start_v + end_v) / 2
            track_width = m * avg_v + b
            track_widths.append(track_width)

        img_threshold_list = []
        for w in track_widths:
            img_threshold = scale * w
            if min_dist_threshold is not None:
                img_threshold = max(img_threshold, min_dist_threshold)
            img_threshold_list.append(img_threshold)

        return img_threshold_list



    def _compute_cost_matrix_single_image(
        self,
        pred_rails: list,
        gt_rails: list,
    ) -> np.ndarray:
        """
        Compute the full pairwise Chamfer distance matrix between predicted
        and GT polylines for one image.

        This is independent of any distance threshold, so callers evaluating
        multiple thresholds should compute it once per image and reuse it
        (see ``evaluate``'s ``_cost_matrix_cache``).

        Returns
        -------
        cost_matrix : np.ndarray of shape (n_pred, n_gt)
        """
        n_pred = len(pred_rails)
        n_gt = len(gt_rails)

        cost = np.empty((n_pred, n_gt))
        for i, pl in enumerate(pred_rails):
            for j, gl in enumerate(gt_rails):
                cost[i, j] = chamfer_distance_polylines(
                    pl, gl, num_points=self.num_sample_points
                )
        return cost

    def _match_single_image(
        self,
        cost: np.ndarray,
        pred_rails: list,
        pred_scores: list,
        threshold: float,
        threshold_list: list = None
    ) -> dict:
        """
        For one image, perform confidence-sorted greedy matching given an
        already-computed Chamfer cost matrix, and return per-prediction
        TP/FP flags.

        This follows the MapTR evaluation protocol:
        - Sort predictions by confidence (descending).
        - For each prediction (in confidence order), match to the closest
          unmatched GT if distance < threshold.

        Returns
        -------
        dict with keys:
        - tp_flags        - list[int] of 0/1 per prediction (confidence-sorted)
        - scores          - list[float] of confidence scores (same order)
        - matched_chamfer - list[float] of Chamfer distances for TP matches
        - cost_matrix     - np.ndarray (n_pred, n_gt) full Chamfer distance matrix
        """
        if self.dist_thr_mode == 'track_width':
            assert threshold_list is not None, "threshold_list must be provided when dist_thr_mode is 'track_width'."
            assert len(threshold_list) == cost.shape[1], "threshold_list length must match number of GT rails."
        else:
            assert isinstance(threshold, (int, float)), "threshold must be a single float for dist_thr_mode 'absolute' or 'relative'."

        n_pred, n_gt = cost.shape

        # Edge cases
        if n_pred == 0:
            return {
                "tp_flags": [],
                "scores": [],
                "matched_chamfer": [],
                "cost_matrix": cost,
            }

        sorted_idx = np.argsort(pred_scores)[::-1]

        if n_gt == 0:
            # All predictions are FP
            output = {
                "tp_flags": [0] * n_pred,
                "scores": [pred_scores[i] for i in sorted_idx],
                "matched_chamfer": [],
                "cost_matrix": cost,
            }
            if self.extended_summary:
                output["pred_rails_sorted"] = [pred_rails[i] for i in sorted_idx]
            return output

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
            dt = threshold if self.dist_thr_mode != 'track_width' else threshold_list[best_gt]

            if best_dist < dt:
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


    def _compute_tp_fp_single_image(
        self,
        img_ident,
        threshold: float,
        min_dist_threshold: Optional[float] = None
    ) -> dict:
        """
        For one image, compute the Chamfer cost matrix and perform
        confidence-sorted greedy matching in one call.

        Kept as a convenience wrapper around ``_compute_cost_matrix_single_image``
        and ``_match_single_image`` for callers (e.g. ``chamfer_viz.py``) that
        need a single-threshold, single-image result without going through
        ``evaluate``'s cost-matrix cache.
        """
        img_name = img_ident

        gt_rails = self.gt[img_name]["rails"].copy()
        pred_rails = self.predictions[img_name]["rails"].copy()
        pred_scores = list(self.predictions[img_name]["score"]).copy()
        pred_rails, pred_scores = self._cap_predictions(pred_rails, pred_scores)
        track_width_line_params = self.gt[img_name].get('track_width_line_parameters', None)
        
        img_threshold = self._resolve_dist_thres(img_name, threshold)
        if self.dist_thr_mode == 'track_width':
            img_threshold_list = self._compute_rail_dist_threshold(gt_rails, track_width_line_params, img_threshold, min_dist_threshold=min_dist_threshold)
        else:
            img_threshold_list = None

        cost = self._compute_cost_matrix_single_image(pred_rails, gt_rails)
        return self._match_single_image(cost, pred_rails, pred_scores, threshold, threshold_list=img_threshold_list)
    
