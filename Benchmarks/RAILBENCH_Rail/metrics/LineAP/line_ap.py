import numpy as np
import copy

import networkx as nx
from networkx.algorithms import bipartite

from Benchmarks.RAILBENCH_Rail.metrics.LineAP.polyline_sampling import point_sampling, segment_sampling
from Benchmarks.RAILBENCH_Rail.utils.ap_utils import calculate_ap_every_point
from Benchmarks.RAILBENCH_Rail.utils.polyline_tools import polyline_orientation
from Benchmarks.RAILBENCH_Rail.utils.ignore_areas import remove_preds_in_ignore
from tqdm import tqdm

from Benchmarks.RAILBENCH_Rail.utils.track_width import track_width_line_parameters

# default line parameters for track width calculation if no valid line can be fitted
M_DEFAULT = 0.52
B_DEFAULT = -520


def _point_to_polyline_distances(gt_points, pred_segments):
    """
    Shortest euclidean distance from each predicted segment (a polyline of
    >=2 vertices, as produced by ``segment_sampling`` - most have 2 vertices,
    but a segment gains an extra vertex whenever an original polyline anchor
    point falls inside it) to each GT point.

    Returns
    -------
    np.ndarray of shape (len(pred_segments), len(gt_points))
    """
    n_pred = len(pred_segments)
    n_gt = len(gt_points)
    if n_pred == 0 or n_gt == 0:
        return np.full((n_pred, n_gt), np.inf)

    P = np.asarray(gt_points, dtype=float)  # (n_gt, 2)

    # Decompose every (possibly multi-vertex) predicted segment into its
    # constituent 2-point sub-segments, remembering which original segment
    # each sub-segment belongs to.
    A_list, B_list, owner = [], [], []
    for seg_idx, seg in enumerate(pred_segments):
        seg = np.asarray(seg, dtype=float)
        A_list.append(seg[:-1])
        B_list.append(seg[1:])
        owner.extend([seg_idx] * (len(seg) - 1))

    A = np.concatenate(A_list, axis=0)  # (M, 2)
    B = np.concatenate(B_list, axis=0)  # (M, 2)
    owner = np.array(owner)

    # Closed-form point-to-segment distance, for all (sub-segment, GT point)
    # pairs at once.
    AB = B - A
    ab_len2 = np.einsum('ij,ij->i', AB, AB)
    AP = P[None, :, :] - A[:, None, :]  # (M, n_gt, 2)
    t = np.einsum('ijk,ik->ij', AP, AB)  # (M, n_gt)
    with np.errstate(invalid='ignore', divide='ignore'):
        t = np.where(ab_len2[:, None] > 0, t / ab_len2[:, None], 0.0)
    t = np.clip(t, 0.0, 1.0)
    proj = A[:, None, :] + t[:, :, None] * AB[:, None, :]  # (M, n_gt, 2)
    sub_dists = np.linalg.norm(P[None, :, :] - proj, axis=-1)  # (M, n_gt)

    # Reduce sub-segment distances back to one row per original predicted
    # segment (shortest distance to any of its sub-segments).
    euclidean_distances = np.full((n_pred, n_gt), np.inf)
    np.minimum.at(euclidean_distances, owner, sub_dists)
    return euclidean_distances


class LineAP:
    def __init__(self, predictions, gt,
                 sample_distance=2, abs_sample_distance_flag=False,
                 dist_thr_mode='track_width',
                 matching_strategy='min_w_maximum_matching', extended_summary=False,
                 max_detections=100,
                 verbose=False):
        """
        
        Parameters
        ----------
        predictions : dict
            Dictionary containing the predicted polylines/rails and their confidence scores.
            Example: {'img01.jpg': 
                        {'rails': [[[u1, v1], [u2, v2], ...], ...],
                        'score': [0.9, 0.8, ...]},
                      'img02.jpg': ...,
                     }

        gt : dict
            Dictionary containing the ground truth polylines/rails in RailBench/COCO format.
            Example: {
                        "images": [
                            {"id": 1, "file_name": <name1.png>, "width": <width>, "height": <height>},
                            {"id": 2, "file_name": <name2.png>, "width": <width>, "height": <height>},
                            ...
                        ],
                        "categories": [
                            {"id": 1, "name": "rail"},
                            {"id": 2, "name": "ignore_area"}
                        ],
                        "annotations": [
                            {"id": <annotation_id>,
                            "image_id": <image_id>,
                            "category_id": <category_id>,
                            "polyline": [[u1, v1], [u2, v2], ...], # if rail 
                            "polygon": [[u1, v1], [u2, v2], ...], # if ignore area
                            "occlusion": <occlusion_level>  # if rail
                            "rightRail": <boolean> # if rail
                            }
                        ]
                     }

        sample_distance : int
            defines distance between sampled points along the polylines. 
        
        abs_sample_distance_flag: bool
            If True, use sample_distance as an absolute distance in pixels for sampling points along the rails. 
            If False, use relative sample distance wrt image width (values are interpreted in percentage). 

        dist_thr_mode : str
            Defines how the distance threshold is interpreted. Default: 'tack_width'.
            Options: 
                1. 'absolute': distance threshold is interpreted as an absolute distance in pixels.
                2. 'relative': distance threshold is interpreted as a relative distance in percentage with respect to the image width.
                3. 'track_width': distance threshold is interpreted as a relative distance in percentage with respect to the track width.

        matching_strategy : str
            Strategy for matching segments. 

            Note: 
            A maximum matching is a matching that contains the largest possible number of edges.
            A maximal matching is a matching that cannot be extended by adding another edge. 
            A maximal matching is not necessarily maximum — it might have fewer edges than the maximum possible matching.

            'maximum_matching': compute the maximum number of matches, and ignores the euclidean distance. (hopcroft_karp_matching)
            'min_w_maximum_matching': compute the maximum number of matches while minimizing the total euclidean distance of all matches. 
                (Among all maximum matchings, finds the one with the smallest possible total edge weight.)
            'min_w_maximal_matching': among all maximal matches, finds the one with the smallest possible total edge weight.

        extended_summary : bool
            If True, the evaluation will return additional information that is needed for plotting functionalities.

        max_detections : int, optional
            If set, only the top ``max_detections`` highest-confidence predicted rails per image are considered
            for matching (analogous to COCO's ``maxDets``). This penalizes over-generation of low-confidence
            predictions: precision is computed only from the kept predictions, while recall is still measured
            against the full, uncapped set of GT points. If None (default), no cap is applied.

        verbose: bool
            If True, print additional information during evaluation.
        """
        if abs_sample_distance_flag and sample_distance > 0:
            self.sample_distance = int(sample_distance)
            assert self.sample_distance > 5, "Sample distance must be greater than 5 pixels to ensure a sufficient number of sample points along the rails."
            self.abs_sample_distance_flag = True
        else:
            self.abs_sample_distance_flag = False
            self.sample_distance = sample_distance
        if verbose:
            if self.abs_sample_distance_flag:
                print("Sample distance:", self.sample_distance, "pixels")
            else:
                print("Sample distance:", self.sample_distance, "% of image width")

        self.dist_thr_mode = dist_thr_mode
        assert self.dist_thr_mode in ['absolute', 'relative', 'track_width'], "Invalid distance threshold mode. Choose from 'absolute', 'relative', or 'track_width'."
        if verbose:
            print("Distance threshold mode:", self.dist_thr_mode)

        predictions_filtered = remove_preds_in_ignore(predictions, gt)
        self.predictions = self._process_predictions(predictions_filtered)
        self.gt = self._process_gt(gt)
        self._checks()

        assert matching_strategy in ['maximum_matching', 'min_w_maximum_matching', 'min_w_maximal_matching'], "Invalid matching strategy."
        self.matching_strategy = matching_strategy

        assert max_detections is None or max_detections > 0, "max_detections must be None or a positive integer."
        self.max_detections = max_detections

        self.extended_summary = extended_summary
        self.verbose = verbose
        self.results = {}

    # ------------------------------------------------------------------
    # Data preparation
    # ------------------------------------------------------------------

    def _process_gt(self, gt):
        """Processes the GT data from RailBench/COCO format to a format that can be used for evaluation."""

        img_id_name_mapping = {img['id']: img['file_name'] for img in gt['images']}
        img_id_width_mapping = {img['id']: img['width'] for img in gt['images']} 
        

        gt_rails = {}
        for img_id, img_name in img_id_name_mapping.items():
            if self.abs_sample_distance_flag:
                s_d = self.sample_distance
            else:
                s_d = int(img_id_width_mapping[img_id] * self.sample_distance/100)

            gt_rails[img_name] = {'rails': [], 'image_width': img_id_width_mapping[img_id], 'sample_distance': s_d}
            if self.dist_thr_mode == 'track_width':
                gt_rails[img_name]['track_ids'] = []
                gt_rails[img_name]['rightRail'] = []
                gt_rails[img_name]['track_width_line_parameters'] = {'m': None, 'b': None}

        cat_id_name_mapping = {cat['id']: cat['name'] for cat in gt['categories']}
        
        for ann in gt['annotations']:
            img_id = ann['image_id']
            img_name = img_id_name_mapping[img_id]
            cat_id = ann['category_id']
            cat_name = cat_id_name_mapping[cat_id]

            if cat_name == 'rail':
                gt_rails[img_name]['rails'].append(polyline_orientation(ann['polyline']))

                if self.dist_thr_mode == 'track_width':
                    gt_rails[img_name]['track_ids'].append(ann['track_id'])
                    gt_rails[img_name]['rightRail'].append(ann['rightRail'])

        # Compute line parameters for track width calculation if dist_thr_mode is 'track_width'
        if self.dist_thr_mode == 'track_width':
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
        for img_ident in self.gt.keys():
            if img_ident not in self.predictions:
                raise KeyError(f"Image '{img_ident}' present in GT but missing in predictions.")

        # Basic validation of prediction entries
        for img_ident, pred in self.predictions.items():
            if not isinstance(pred, dict):
                raise TypeError(f"Prediction for '{img_ident}' must be a dict with keys 'rails' and 'score'.")
            if 'rails' not in pred or 'score' not in pred:
                raise KeyError(f"Predictions for '{img_ident}' must contain 'rails' and 'score' keys.")
            if not isinstance(pred['rails'], list) or not isinstance(pred['score'], list):
                raise TypeError(f"'rails' and 'score' for '{img_ident}' must be lists.")
            if len(pred['rails']) != len(pred['score']):
                raise ValueError(f"Number of predicted rails and score values mismatch for '{img_ident}': {len(pred['rails'])} vs {len(pred['score'])}.")
            for rail in pred['rails']:
                if not isinstance(rail, list) or len(rail) < 2:
                    raise ValueError(f"Each predicted rail for '{img_ident}' must be a list of at least 2 points.")
                for point in rail:
                    if not isinstance(point, list) or len(point) != 2:
                        raise ValueError(f"Each point in the predicted rails for '{img_ident}' must be a list of 2 coordinates [u, v].")

    # ------------------------------------------------------------------
    # Evaluation functions
    # ------------------------------------------------------------------

    def evaluate(self, dist_thresholds=[5], min_dist_threshold=5, orient_threshold=10):
        """
        Run evaluation for specified thresholds. 

        Parameters
        ----------
        dist_thresholds : list of int or floats
            List of distance thresholds for matching. 
            Depending on the dist_thr_mode these are either interpreted as absolute distances in pixels or relative distances (in percentage) with respect to the image width or track width.
        min_dist_threshold : int or float
            Minimum distance threshold for matching. Only used if dist_thr_mode is 'track_width'.
        orient_threshold: int 
            Maximum orientation difference in degrees for matching.
        """
        if self.verbose:
            print("Reset results and start evaluation...")
        self.results = {}
        self._frame_prep_cache = {}

        ap_list = []
        for d_t in dist_thresholds:
            if self.verbose:
                if self.dist_thr_mode == 'absolute':
                    print(f"Evaluating for distance threshold = {d_t} px ...")
                elif self.dist_thr_mode == 'relative':
                    print(f"Evaluating for distance threshold wrt image width = {d_t} % ...")
                else:
                    print(f"Evaluating for distance threshold wrt to track width = {d_t} % ...")

            result_key = f"dist_thres_{d_t}" if self.dist_thr_mode == 'absolute' else f"rel_dist_thres_{d_t}"

            self.results[result_key] = dict()
            if self.extended_summary:
                self.results[result_key]['images'] = dict()
            # compute TP and FP and total number of rails in gt
            tp, fp, n_gt, avg_match_dist = self.compute_tp_fp(
                                            predictions = self.predictions, 
                                            gt = self.gt,
                                            dist_thres = d_t, 
                                            min_dist_threshold = min_dist_threshold,
                                            orient_thres=orient_threshold,
                                            result_key=result_key)

            # compute AP 
            acc_FP = np.cumsum(fp)
            acc_TP = np.cumsum(tp)
            rec = acc_TP / n_gt
            prec = np.divide(acc_TP, (acc_FP + acc_TP))

            [ap, mpre, mrec, ii] = calculate_ap_every_point(rec, prec)

            ap_list.append(ap)
            self.results[result_key]['AP'] = ap
            self.results[result_key]['avg_match_dist'] = avg_match_dist

        # compute mAP (mean across all distance thresholds)
        mAP = np.mean(ap_list) if len(ap_list) > 0 else 0.0
        self.results['mAP'] = mAP

        return self.results
    

    def compute_tp_fp(self, predictions, gt, dist_thres=10, min_dist_threshold=5, orient_thres=10, result_key=None):
        """
        Computes true positives (TP) and false positives (FP) for the given predictions and ground truth data. 

        Parameters
        ----------
        predictions : dict
            Dictionary containing the predicted polylines/rails and their confidence scores.
            Example: {'img01.jpg': 
                            {'rails': [[[u1, v1], [u2, v2], ...], ...],
                            'score': [0.9, 0.8, ...]},
                    'img02.jpg': ...
                        }

        gt : dict
            Dictionary containing the ground truth polylines/rails and ignore areas.
            Example: {'img01.jpg': 
                            {'rails': [[[u1, v1], [u2, v2], ...], ...],
                            'ignore_areas': [[[u1, v1], [u2, v2], ...], ...]},
                    'img02.jpg': ...
                    }

        dist_thres : int
            (relative) distance threshold for matching segments in pixels. Default: 10 pixels.

        min_dist_threshold : int
            Minimum distance threshold for matching segments in pixels. Only used if dist_thr_mode is 'track_width'. Default: 5 pixels.

        orient_thres : int
            Orientation threshold for matching segments. Default: 10.

        Returns
        -------
        tp : np.ndarray
            Array of true positives for each predicted segment, sorted according to predicted confidence (primary) and whether the sample is a tp (secondary).
        fp : np.ndarray
            Array of false positives for each predicted segment. Complement to tp. 
        n_gt : int
            Total number of ground truth points.
        avg_matching_distance : float
            Average distance of matched segments. If no matches were found, returns -1.0.
        """
        
        first_image = True

        matching_distance = []
        
        # iterate over each image in predictions and gt
        for i, img_ident in enumerate(tqdm(gt.keys(), desc="Images", unit="img", disable=not self.verbose)):
            if img_ident not in predictions:
                print(f"Image {img_ident} not found in predictions.")
                continue

            img_dist_thres = self._resolve_dist_thres(img_ident, dist_thres)

            sample_distance = gt[img_ident]['sample_distance']  

            gt_rails = gt[img_ident]['rails'].copy()
            pred_rails = predictions[img_ident]['rails'].copy()
            pred_confidence = predictions[img_ident]['score'].copy()
            track_width_line_params = gt[img_ident].get('track_width_line_parameters', {'m': None, 'b': None})

            if img_ident not in self._frame_prep_cache:
                self._frame_prep_cache[img_ident] = self._prepare_frame(
                    pred_rails=pred_rails, pred_confidence=pred_confidence,
                    gt_rails=gt_rails, sample_distance=sample_distance, 
                    track_width_line_params=track_width_line_params)
            prep = self._frame_prep_cache[img_ident]

            output = self._match_frame(prep, dist_thres=img_dist_thres, orient_thres=orient_thres, min_dist_threshold=min_dist_threshold)
            true_positives = output['true_positives']
            n_gt_pts = output['n_gt_pts']

            if output['avg_matching_distance'] >= 0:
                matching_distance.append(output['avg_matching_distance'])

            if first_image:
                tp = true_positives
                n_gt = n_gt_pts
                scores = list(output['pred_confidence'])

                first_image = False
            else:
                tp = np.concatenate((tp, true_positives))
                n_gt += n_gt_pts
                scores.extend(output['pred_confidence'])

            if self.extended_summary:
                self.results[result_key]['images'][img_ident] = output

        # sort true_positives according to scores (primary) and is_true_positive (secondary)
        sorted_indices = np.lexsort((tp, scores))[::-1]
        tp = tp[sorted_indices]
        scores = [scores[i] for i in sorted_indices]

        fp = np.ones((len(tp))) - tp

        if len(matching_distance) > 0:
            avg_matching_distance = np.mean(matching_distance)
        else:
            avg_matching_distance = -1.0

        return tp, fp, n_gt, avg_matching_distance

    def _resolve_dist_thres(self, img_ident, dist_thres):
        if self.dist_thr_mode == 'absolute':
            return dist_thres
        elif self.dist_thr_mode == 'relative':
            return int(dist_thres/100.0 * self.gt[img_ident]['image_width'])
        else:  # 'track_width'
            # Transform percentage into scale factor
            return dist_thres/100.0

    def compute_tp_fp_single_frame(self, img_ident, dist_thres=10, orient_thres=10, min_dist_threshold=5):

        pred_rails = self.predictions[img_ident]['rails'].copy()
        pred_confidence = self.predictions[img_ident]['score'].copy()
        gt_rails = self.gt[img_ident]['rails'].copy()
        sample_distance = self.gt[img_ident]['sample_distance']
        track_width_line_params = self.gt[img_ident].get('track_width_line_parameters', {'m': M_DEFAULT, 'b': B_DEFAULT})

        img_dist_thres = self._resolve_dist_thres(img_ident, dist_thres)
        
        prep = self._prepare_frame(pred_rails, pred_confidence, gt_rails, sample_distance, track_width_line_params=track_width_line_params)
        return self._match_frame(prep, dist_thres=img_dist_thres, orient_thres=orient_thres, min_dist_threshold=min_dist_threshold)

    def _prepare_frame(self, pred_rails, pred_confidence, gt_rails, sample_distance=50, track_width_line_params=None):
        """
        Threshold-independent preparation for one frame: sorts predictions by
        confidence, samples GT/predictions into points/segments, filters
        segments in ignore areas, and (in the general case) computes the
        pairwise distance/orientation matrices used for matching. None of
        this depends on dist_thres/orient_thres, so it can be computed once
        per frame and reused across all thresholds evaluated for it.

        Returns a dict with a 'case' key ('empty_pred', 'empty_gt', or
        'full') plus whatever data ``_match_frame`` needs for that case.
        """
        if len(pred_rails) > 0:
            # sort predictions according to predicted confidence (high to low)
            sorted_indices = np.argsort(pred_confidence)[::-1]
            pred_rails = [pred_rails[i] for i in sorted_indices]
            pred_confidence = [pred_confidence[i] for i in sorted_indices]

            if self.max_detections is not None:
                # Keep only the top-`max_detections` highest-confidence predicted rails
                # (analogous to COCO's maxDets), to penalize over-generation.
                pred_rails = pred_rails[:self.max_detections]
                pred_confidence = pred_confidence[:self.max_detections]

        # sampling gt
        if len(gt_rails) > 0:
            gt_points, gt_orient, _ = point_sampling(gt_rails, sample_distance, midpoints=True)
            if self.extended_summary:
                gt_segments, _, _ = segment_sampling(gt_rails, sample_distance=sample_distance)
            if self.dist_thr_mode == 'track_width':
                assert track_width_line_params is not None, "track_width_line_params must be provided when dist_thr_mode is 'track_width'."
                # Compute track width for each GT point based on the line parameters
                m = track_width_line_params['m']
                b = track_width_line_params['b']

                gt_track_widths = []
                for point in gt_points:
                    u, v = point
                    track_width = m * v + b  
                    gt_track_widths.append(track_width)


        # special cases: no gt rails, no pred rails, etc.
        if len(pred_rails) == 0:
            # no predictions, return empty results
            return {
                'case': 'empty_pred',
                'gt_points': gt_points if 'gt_points' in locals() else [],
                'gt_segments': gt_segments if 'gt_segments' in locals() else [],
                'gt_track_width': gt_track_widths if 'gt_track_widths' in locals() else None,
            }

        elif len(gt_rails) == 0:
            # no ground truth, all predictions are false positives regardless of threshold
            return {
                'case': 'empty_gt',
                'pred_rails': pred_rails,
                'pred_confidence': pred_confidence,
            }

        # sampling predictions
        pred_segments, pred_orient, rail_index_list = segment_sampling(pred_rails, sample_distance=sample_distance)
        pred_confidence_extended = np.zeros((len(pred_segments)))
        for i, rail_index in enumerate(rail_index_list):
            pred_confidence_extended[i] = pred_confidence[rail_index]
        pred_confidence = pred_confidence_extended

        euclidean_distances, orient_differences = self._compute_distance_orient_matrices(
            pred_segments=pred_segments, pred_orient=pred_orient,
            gt_points=gt_points, gt_orient=gt_orient)

        return {
            'case': 'full',
            'pred_segments': pred_segments,
            'pred_confidence': pred_confidence,
            'gt_points': gt_points,
            'gt_segments': gt_segments if self.extended_summary else [],
            'euclidean_distances': euclidean_distances,
            'orient_differences': orient_differences,
            'gt_track_widths': gt_track_widths if self.dist_thr_mode == 'track_width' else None
        }

    def _match_frame(self, prep, dist_thres, orient_thres, min_dist_threshold):
        """
        Match line segments. 

        Args:
            prep: dict
                Output of _prepare_frame() for a single frame.
            dist_thres: float
                Distance threshold for matching segments. Note, that in modes 'absolute' and 'relative', this is an absolute distance in pixels, while in mode 'track_width', this is a scale factor that is multiplied with the track width to derive a per-GT-point distance threshold.
            orient_thres: float
                Orientation threshold for matching segments.
            min_dist_threshold: float
                Minimum distance threshold for matching segments. Only used if self.dist_thr_mode is 'track_width'.
        """
        if prep['case'] == 'empty_pred':
            gt_points = prep['gt_points']
            output = {
                'true_positives': np.zeros((0)),
                'n_gt_pts': len(gt_points),
                'avg_matching_distance': -1,
                'pred_confidence': []
            }
            if self.extended_summary:
                output['pred_segments'] = []
                output['gt_points'] = gt_points
                output['gt_segments'] = prep['gt_segments']
                output['matched_gt_points'] = np.zeros((len(gt_points)))
                output['gt_track_width'] = prep['gt_track_width']
            return output

        if prep['case'] == 'empty_gt':
            pred_rails = prep['pred_rails']
            pred_confidence = prep['pred_confidence']
            output = {
                'true_positives': np.zeros((len(pred_rails))),
                'n_gt_pts': 0,
                'avg_matching_distance': -1,
                'pred_confidence': pred_confidence
            }
            if self.extended_summary:
                output['pred_segments'] = pred_rails
                output['gt_points'] = []
                output['gt_segments'] = []
                output['matched_gt_points'] = np.zeros((0))
                output['gt_track_width'] = None
            return output

        # case == 'full'
        pred_segments = prep['pred_segments']
        pred_confidence = prep['pred_confidence']
        gt_points = prep['gt_points']
        gt_track_widths = prep['gt_track_widths']

        # perform matching
        match_pred_ind, match_gt_ind, avg_matching_distance, graphs = self._match_from_matrices(
            euclidean_distances=prep['euclidean_distances'],
            orient_differences=prep['orient_differences'],
            pred_confidence=pred_confidence,
            dist_thres=dist_thres, orient_thres=orient_thres, track_widths=gt_track_widths, min_dist_threshold=min_dist_threshold)

        true_positives = np.zeros((len(pred_segments)))
        true_positives[match_pred_ind] = 1

        n_gt_pts = len(gt_points)

        if self.extended_summary:
            matched_gt_points = np.zeros((len(gt_points)))
            matched_gt_points[match_gt_ind] = 1

        output = {
            'true_positives': true_positives,
            'n_gt_pts': n_gt_pts,
            'avg_matching_distance': avg_matching_distance,
            'pred_confidence': pred_confidence
        }

        if self.extended_summary:
            output['pred_segments'] = pred_segments
            output['gt_points'] = gt_points
            output['gt_segments'] = prep['gt_segments']

            output['matched_gt_points'] = matched_gt_points

            output['graphs'] = graphs

        return output
    

    def _compute_distance_orient_matrices(self, pred_segments, pred_orient, gt_points, gt_orient):
        """
        Computes the pairwise euclidean distance matrix (predicted segments
        vs. GT points) and the pairwise orientation-difference matrix.
        Independent of dist_thres/orient_thres, so callers evaluating
        multiple thresholds should compute this once per frame and reuse it
        (see ``_prepare_frame``'s cached output).
        """
        # compute shortest distances between GT points and predicted segments
        euclidean_distances = _point_to_polyline_distances(gt_points, pred_segments)

        # compute angular distances of GT segments and predicted segments
        orient_differences = np.full((len(pred_orient), len(gt_orient)), np.inf)
        gt_orient = np.array(gt_orient)
        for idx_pred, o in enumerate(pred_orient):
            a = np.abs(gt_orient - o)
            b = np.abs(np.abs(gt_orient - o) - 360)
            orient_differences[idx_pred] = np.min(np.stack((a, b)), axis=0)

        return euclidean_distances, orient_differences

    def _match_from_matrices(self, euclidean_distances, orient_differences, pred_confidence, dist_thres, orient_thres, track_widths=None, min_dist_threshold=5):
        """
        Threshold-dependent matching step given precomputed distance/
        orientation matrices. ``confidence_matching`` mutates its inputs
        in place, so both matrices are copied here to keep a cached
        ``euclidean_distances``/``orient_differences`` pair reusable across
        multiple threshold evaluations.
        """
        matches, graphs = self.confidence_matching( cost_matrix = euclidean_distances.copy(),
                                                    orient_matrix = orient_differences.copy(),
                                                    pred_confidence = pred_confidence,
                                                    dist_thres = dist_thres,
                                                    orient_thres = orient_thres, 
                                                    track_widths = track_widths,
                                                    min_dist_threshold = min_dist_threshold)

        if len(matches) == 0:
            return [], [], -1, graphs

        match_pred_ind, match_gt_ind = zip(*matches)
        match_pred_ind = list(match_pred_ind)
        match_gt_ind = list(match_gt_ind)

        total_costs = 0
        for i, j in matches:
            total_costs += euclidean_distances[i, j]
        avg_matching_distance = total_costs / len(matches)

        return match_pred_ind, match_gt_ind, avg_matching_distance, graphs


    def confidence_matching(self, cost_matrix, orient_matrix, pred_confidence, dist_thres, orient_thres, track_widths=None, min_dist_threshold=5):
        """
        Computes the matches iteratively starting with the predictions with the highest confidence score.   

        Parameters
        ----------
        cost_matrix : np.ndarray
            2D array of shape (num_pred_segments, num_gt_points) containing the euclidean distances between predicted segments and GT points.
        orient_matrix : np.ndarray
            2D array of shape (num_pred_segments, num_gt_points) containing the orientation differences between predicted segments and GT points.
        pred_confidence : list
            List of confidence scores for the predicted segments. Must be sorted in descending order.
        dist_thres : float
            Distance threshold for the matching.
        orient_thres : float
            Orientation threshold for the matching.
        track_widths : list, optional
            List of track widths for the GT points. Required if self.dist_thr_mode is 'track_width'.

        Returns
        -------
        all_matches : list of tuples
            List of matched pairs (pred_index, gt_index).
            Each pair indicates that the predicted segment at pred_index is matched to the GT point at gt_index.
            If no matches are found, returns an empty list.
        graphs : dict
            If self.extended_summary is False, returns an empty dict.
            keys: confidence scores
            values: list of all edges in the graph as tuples (pred_index, gt_index, cost, orient, is_match).
                Each tuple indicates that there is an edge between the predicted segment at pred_index and the GT point at gt_index with the given cost and orientation difference.
                If no edges are found for a confidence score, the value is an empty list.
        """

        all_matches = []
        graphs = {}

        # iterate over all confidence scores and compute the matches
        unique_confidences = np.sort(np.unique(pred_confidence))[::-1]

        for c in unique_confidences:
            # get the indices of the segments with the current confidence score
            indices = np.where(pred_confidence == c)[0]

            # filter the cost matrix and orientation matrix for the current confidence score
            cost_matrix_filtered = cost_matrix[indices, :]
            orient_matrix_filtered = orient_matrix[indices, :]

            # compute the matches for the current confidence score
            matches, graph_summary = self.graph_matching(cost_matrix=cost_matrix_filtered,
                                                         orient_matrix=orient_matrix_filtered,
                                                         dist_thres=dist_thres,
                                                         orient_thres=orient_thres, 
                                                         track_widths=track_widths,
                                                         min_dist_threshold=min_dist_threshold)
            # !!! Note: the indices in matches for the predictions correspond to the filtered cost matrix !!!

            if len(matches) > 0:
                match_pred_ind_filtered, match_gt_ind = zip(*matches)
                # map indices back to original indices
                match_pred_ind = list(indices[np.array(match_pred_ind_filtered)])

                all_matches.extend(zip(match_pred_ind, match_gt_ind))

                if self.extended_summary:
                    graph_summary = [(indices[i], j, cost, orient, ((indices[i], j) in all_matches)) for i, j, cost, orient in graph_summary]

                    graphs[c] = graph_summary


                # set the columns of the matched ground truth points from the cost matrix and orientation matrix to inf (to avoid matching them again)
                cost_matrix[:, match_gt_ind] = np.inf
                orient_matrix[:, match_gt_ind] = np.inf

        return all_matches, graphs


    def graph_matching(self, cost_matrix, orient_matrix, dist_thres, orient_thres, track_widths=None, min_dist_threshold=5):
        """
        Builds a bipartite graph and computes matches. 
        Expects that each row in the cost_matrix and orient_matrix corresponds to a predicted segment and each column corresponds to a GT point.

        Graph: 
        If the entries (i,j) in the cost_matrix and the orient_matrix are below the corresponding thresholds, an edge is added between the nodes i and j.

        Returns
        -------
        matches : list of tuples
            List of matched pairs (pred_index, gt_index).
            Each pair indicates that the predicted segment at pred_index is matched to the GT point at gt_index.
            If no matches are found, returns an empty list.
        graph_summary : list of tuples
            Empty if self.extended_summary is False.
            List of all edges in the graph as tuples (pred_index, gt_index, cost, orient).
            Each tuple indicates that there is an edge between the predicted segment at pred_index and the GT point at gt_index with the given cost and orientation difference.
            If no edges are found, returns an empty list.
        """

        if self.dist_thr_mode == 'track_width':
            assert len(track_widths) == cost_matrix.shape[1], "Length of track_widths must match the number of GT points (columns in cost_matrix)."

        G = nx.Graph()

        graph_summary = []

        plotting = False

        if self.matching_strategy == 'maximum_matching' or plotting:
            left_set = set()

        # Determine the valid-edge mask (cost below threshold and orientation below
        # threshold) for the whole matrix at once instead of a per-cell Python loop --
        # this is the dominant cost of this function otherwise, since it's called once
        # per (image, threshold, confidence-group).
        if self.dist_thr_mode == 'track_width':
            dt = dist_thres * np.asarray(track_widths, dtype=float)  # per GT point (column)
            np.maximum(dt, min_dist_threshold, out=dt)
            valid = (cost_matrix <= dt[np.newaxis, :]) & (orient_matrix <= orient_thres)
        else:
            valid = (cost_matrix <= dist_thres) & (orient_matrix <= orient_thres)

        # np.nonzero visits entries in row-major order, i.e. the same (i, j) order the
        # previous nested loop produced.
        rows, cols = np.nonzero(valid)
        rows = rows.tolist()
        cols = cols.tolist()
        weights = cost_matrix[valid].tolist()

        G.add_weighted_edges_from(zip((f"row_{i}" for i in rows), (f"col_{j}" for j in cols), weights))

        if self.extended_summary:
            orients = orient_matrix[valid].tolist()
            graph_summary = list(zip(rows, cols, weights, orients))

        if self.matching_strategy == 'maximum_matching' or plotting:
            left_set = {f"row_{i}" for i in rows}


        if plotting:  # visualize graph
            import matplotlib.pyplot as plt

            nx.draw(G, with_labels=True, pos=nx.bipartite_layout(G, left_set))

            pos = nx.spring_layout(G, seed=7)
            nx.draw_networkx_nodes(G, pos, node_size=300)
            edges = [(u, v) for (u, v, d) in G.edges(data=True)]
            nx.draw_networkx_edges(G, pos, edgelist=edges, width=6)
            nx.draw_networkx_labels(G, pos, font_size=20, font_family="sans-serif")

            ax = plt.gca()
            ax.margins(0.08)
            plt.axis("off")
            plt.tight_layout()
            plt.show()


        if self.matching_strategy == 'maximum_matching':
            matching = bipartite.hopcroft_karp_matching(G, left_set)
            matches = [(int(r[4:]), int(c[4:])) if r.startswith("row") else (int(c[4:]), int(r[4:])) 
                for r, c in matching.items() if r.startswith("row")]
            
        elif self.matching_strategy == 'min_w_maximum_matching':
            matching = nx.min_weight_matching(G)
            matches = [(int(r[4:]), int(c[4:])) if r.startswith("row") else (int(c[4:]), int(r[4:])) for r, c in matching]

        elif self.matching_strategy == 'min_w_maximal_matching':
            matching = self.minimal_weight_maximal_matching(G)
            matches = [(int(r[4:]), int(c[4:])) if r.startswith("row") else (int(c[4:]), int(r[4:])) for r, c in matching]

        return matches, graph_summary

    @staticmethod
    def minimal_weight_maximal_matching(G):
        """
        Computes the minimum-weight maximal matching of a graph.
        """
        
        # Sort edges by weight (ascending order)
        sorted_edges = sorted(G.edges(data=True), key=lambda x: x[2]['weight'])

        matching = set()
        matched_nodes = set()

        # Greedily select edges ensuring maximality with minimal weight
        for u, v, _ in sorted_edges:
            if u not in matched_nodes and v not in matched_nodes:
                matching.add((u, v))
                matched_nodes.add(u)
                matched_nodes.add(v)

        return matching


    # ------------------------------------------------------------------
    # Summary and results functions
    # ------------------------------------------------------------------

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
        print("LineAP Results")
        print("=" * 50)
        for key, res in self.results.items():
            if key == "mAP":
                continue
            print(f"  {self._format_threshold_label(key)}:")
            print(f"    AP                    = {res['AP']:.4f}")
            print(f"    Avg. matching dist.   = {res['avg_match_dist']:.4f}")
        if "mAP" in self.results:
            print("-" * 50)
            print(f"  mAP (across thresholds) = {self.results['mAP']:.4f}")
        print("=" * 50)

    def return_results(self):
        """
        Returns the evaluation results as a dictionary.
        """
        return self.results

