from matplotlib import pyplot as plt
import numpy as np
import os

from shapely import LineString, Point
import networkx as nx
from networkx.algorithms import bipartite

from Benchmarks.RAILBENCH_Rail.metrics.LineAP.polyline_sampling import point_sampling, segment_sampling, segment_in_ignore
from Benchmarks.RAILBENCH_Rail.metrics.LineAP.visualizations import visualize_segmentwise_evaluation
from tqdm import tqdm

class LineAP:
    def __init__(self, predictions, gt, 
                 sample_distance=50, abs_sample_distance_flag=True,
                 matching_strategy='min_w_maximum_matching', extended_summary=False):
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
        """
        if abs_sample_distance_flag and sample_distance > 0:
            self.sample_distance = int(sample_distance)
            assert self.sample_distance > 5, "Sample distance must be greater than 5 pixels to ensure a sufficient number of sample points along the rails."
            self.abs_sample_distance_flag = True
        else:
            self.abs_sample_distance_flag = False
            self.sample_distance = sample_distance

        self.predictions = self._process_predictions(predictions)
        self.gt = self._process_gt(gt)
        self._checks()

        assert matching_strategy in ['maximum_matching', 'min_w_maximum_matching', 'min_w_maximal_matching'], "Invalid matching strategy."
        self.matching_strategy = matching_strategy

        self.extended_summary = extended_summary
        self.results = {}

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

            gt_rails[img_name] = {'rails': [], 'ignore_areas': [], 'image_width': img_id_width_mapping[img_id], 'sample_distance': s_d}

        cat_id_name_mapping = {cat['id']: cat['name'] for cat in gt['categories']}
        
        for ann in gt['annotations']:
            img_id = ann['image_id']
            img_name = img_id_name_mapping[img_id]
            cat_id = ann['category_id']
            cat_name = cat_id_name_mapping[cat_id]

            if cat_name == 'rail':
                gt_rails[img_name]['rails'].append(ann['polyline'])
            elif cat_name == 'ignore_area':
                gt_rails[img_name]['ignore_areas'].append(ann['polygon'])

        return gt_rails

    def _process_predictions(self, predictions):
        """
        Predicted polylines should start in the foreground and end in the background. 
        This function checks the orientation of the predicted polylines and flips them if necessary.
        """
        for img_ident, pred in predictions.items():
            for i, rail in enumerate(pred['rails']):
                if len(rail) < 2:
                    continue
                start_point_v = rail[0][1]
                end_point_v = rail[-1][1]

                if start_point_v < end_point_v:
                    pred['rails'][i] = rail[::-1]

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

    def evaluate(self, dist_thresholds=[10], use_abs_dist=True, orient_threshold=10):
        """
        Run evaluation for specified thresholds. 

        Parameters
        ----------
        dist_thresholds : list of int
            List of distance thresholds for matching. 
            Depending on the use_abs_dist flag these are either interpreted as absolute distances in pixels or relative distances (in percentage) with respect to the image width.
        use_abs_dist : bool
            Decides whether to use absolute distance thresholds or relative distance thresholds for evaluation.

        orient_threshold: int 
            Maximum orientation difference in degrees for matching.
        """

        print("Reset results and start evaluation...")
        self.results = {}

        if use_abs_dist:
            print("Evaluating for absolute distance thresholds.")
        else:
            print("Evaluating for relative distance thresholds.")


        ap_list = []
        for d_t in dist_thresholds:
            if use_abs_dist:
                print(f"Evaluating for distance threshold = {d_t} px ...")
            else:
                print(f"Evaluating for distance threshold = {d_t} % ...")
            result_key = f"dist_thres_{d_t}" if use_abs_dist else f"rel_dist_thres_{d_t}"

            self.results[result_key] = dict()
            if self.extended_summary:
                self.results[result_key]['images'] = dict()
            # compute TP and FP and total number of rails in gt
            tp, fp, n_gt, avg_match_dist = self.compute_tp_fp(
                                            predictions = self.predictions, 
                                            gt = self.gt,
                                            dist_thres = d_t, 
                                            abs_dist_flag=use_abs_dist, 
                                            orient_thres=orient_threshold,
                                            result_key=result_key)

            # compute AP 
            acc_FP = np.cumsum(fp)
            acc_TP = np.cumsum(tp)
            rec = acc_TP / n_gt
            prec = np.divide(acc_TP, (acc_FP + acc_TP))

            [ap, mpre, mrec, ii] = self.calculate_ap_every_point(rec, prec)

            ap_list.append(ap)
            self.results[result_key]['AP'] = ap
            self.results[result_key]['avg_match_dist'] = avg_match_dist

        # compute mAP (mean across all distance thresholds)
        mAP = np.mean(ap_list) if len(ap_list) > 0 else 0.0
        self.results['mAP'] = mAP

        return self.results
    

    def print_summary(self):
        """
        Prints a summary of the evaluation results.
        """
        for k, res in self.results.items():
            if k == 'mAP':
                print(f"mAP: {res:.2f}")
                print("-----")
            else:
                if k.startswith("rel_dist_thres_"):
                    d_t = float(k.split("_")[-1])
                    if d_t < 1:
                        print(f"Relative distance threshold: {d_t}")
                    else:
                        print(f"Distance threshold: {d_t} px")

                print(f"AP: {res['AP']:.2f}")
                print(f"Average matching distance: {res['avg_match_dist']:.2f}")
            print("-----")

    def return_results(self):
        """
        Returns the evaluation results as a dictionary.
        """
        return self.results


    def compute_tp_fp(self, predictions, gt, dist_thres=10, abs_dist_flag=True, orient_thres=10, result_key=None):
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

        abs_dist_flag : bool
            If True, dist_thres is treated as an absolute distance threshold in pixels.
            If False, dist_thres is treated as a relative distance threshold, and the actual distance threshold is computed as dist_thres * image_width for each image.

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
        for i, img_ident in enumerate(tqdm(gt.keys(), desc="Images", unit="img")):
            if img_ident not in predictions:
                print(f"Image {img_ident} not found in predictions.")
                continue

            if not abs_dist_flag:
                img_dist_thres = int(dist_thres/100 * gt[img_ident]['image_width'])
            else:
                img_dist_thres = dist_thres

            sample_distance = gt[img_ident]['sample_distance']  

            gt_rails = gt[img_ident]['rails'].copy()
            ignore_areas = gt[img_ident]['ignore_areas'].copy()
            pred_rails = predictions[img_ident]['rails'].copy()
            pred_confidence = predictions[img_ident]['score'].copy()

            output = self.compute_tp_fp_single_frame(pred_rails=pred_rails, pred_confidence=pred_confidence,
                                                     gt_rails=gt_rails, ignore_areas=ignore_areas, 
                                                     sample_distance=sample_distance,
                                                     dist_thres=img_dist_thres, orient_thres=orient_thres)
            true_positives = output['true_positives']
            n_gt_pts = output['n_gt_pts']

            if output['avg_matching_distance'] >= 0:
                matching_distance.append(output['avg_matching_distance'])

            if first_image:
                tp = true_positives
                #fp = np.ones((len(true_positives))) - tp
                n_gt = n_gt_pts
                #matched_gt = output['matched_gt_points']
                scores = output['pred_confidence']

                first_image = False
            else:
                tp = np.concatenate((tp, true_positives))
                #fp = np.concatenate((fp, np.ones((len(true_positives))) - true_positives))
                n_gt += n_gt_pts
                scores.extend(output['pred_confidence'])
                #matched_gt = np.concatenate((matched_gt, matched_gt_points))

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


    def compute_tp_fp_single_frame(self, pred_rails, pred_confidence, gt_rails, ignore_areas, sample_distance=50, dist_thres=10, orient_thres=10):
        """
        Computes TP and FP for a single frame using an AP inspired strategy. 

        Predictions and Ground truth are divided into small polyline segments. Predictions are sorted according to predicted confidence (called 'score' in AP). 
        Matching algorithm starts with predicted segments with highest confidence and ends with lowest confidence. Within each predicted confidence value step, 
        the matching is performed based on the specified matching_strategy (e.g. 'min_w_maximum_matching'). Each prediction and ground truth segment can be matched
        at most once. 

        Inputs: 
            pred_rails: list of predicted polylines/rails, each polyline is a list of points [[u1, v1], [u2, v2], ...]
            pred_confidence: list of confidence scores for each predicted polyline
            gt_rails: list of ground truth polylines/rails, each polyline is a list of points [[u1, v1], [u2, v2], ...]
            ignore_areas: list of ignore areas, each area is a list of points [[u1, v1], [u2, v2], ...]
            sample_distance: distance between sampled points along the polylines
            dist_thres: distance threshold for matching segments
            orient_thres: orientation threshold for matching segments
        
        Returns:
            output: dictionary containing the following keys:
                - true_positives: array of true positives for each predicted segment (sorted according to predicted confidence)
                - n_gt_pts: total number of ground truth points
                - avg_matching_distance: average distance of matched segments
                - pred_confidence: list of confidence scores for each predicted segment (sorted according to predicted confidence)

            If self.extended_summary is True, the following keys are also included:
                - pred_segments: list of predicted segments (sorted according to predicted confidence)
                - gt_points: list of ground truth points
                - gt_segments: list of ground truth segments
                - matched_gt_points: array of matched ground truth points 
                - graphes: dict of graph information with confidence values as keys and list of edges as values
        """

        if len(pred_rails) > 0:
            # sort predictions according to predicted confidence (high to low)
            sorted_indices = np.argsort(pred_confidence)[::-1]
            pred_rails = [pred_rails[i] for i in sorted_indices]
            pred_confidence = [pred_confidence[i] for i in sorted_indices]
        
        # sampling gt
        if len(gt_rails) > 0:
            gt_points, gt_orient, _ = point_sampling(gt_rails, sample_distance, midpoints=True)
            if self.extended_summary:
                gt_segments, _, _ = segment_sampling(gt_rails, sample_distance=sample_distance)


        # special cases: no gt rails, no pred rails, etc.
        if len(pred_rails) == 0:
            # no predictions, return empty results
            output = {
                'true_positives': np.zeros((0)),
                'n_gt_pts': len(gt_points) if 'gt_points' in locals() else 0,  
                'avg_matching_distance': -1,
                'pred_confidence': []
            }
            if self.extended_summary:
                output['pred_segments'] = []
                output['gt_points'] = gt_points if 'gt_points' in locals() else []
                output['gt_segments'] = gt_segments if 'gt_segments' in locals() else []
                output['matched_gt_points'] = np.zeros((len(gt_points)))
            return output
        
        elif len(gt_rails) == 0:
            # no ground truth, return all predictions as false positives
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
            return output



        # sampling predictions
        pred_segments, pred_orient, rail_index_list = segment_sampling(pred_rails, sample_distance=sample_distance)
        pred_confidence_extended = np.zeros((len(pred_segments)))
        for i, rail_index in enumerate(rail_index_list):
            pred_confidence_extended[i] = pred_confidence[rail_index]
        pred_confidence = pred_confidence_extended

        # filter out predicted segments in ignore areas
        midpoint_in_ignore = segment_in_ignore(pred_rails, ignore_areas, sample_distance)
        pred_segments = [seg for seg, ignore in zip(pred_segments, midpoint_in_ignore) if not ignore]
        pred_orient = [orient for orient, ignore in zip(pred_orient, midpoint_in_ignore) if not ignore]
        pred_confidence = [conf for conf, ignore in zip(pred_confidence, midpoint_in_ignore) if not ignore]

        # perform matching
        match_pred_ind, match_gt_ind, avg_matching_distance, graphs = self.point_segment_matching(
            pred_segments=pred_segments, pred_confidence=pred_confidence, pred_orient=pred_orient, 
            gt_points=gt_points, gt_orient=gt_orient,
            dist_thres=dist_thres, orient_thres=orient_thres)
        
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
            output['gt_segments'] = gt_segments

            output['matched_gt_points'] = matched_gt_points

            output['graphs'] = graphs

        return output
    

    def point_segment_matching(self, pred_segments, pred_confidence, pred_orient, gt_points, gt_orient, dist_thres, orient_thres):
        """
        Takes a set of gt points and a set of predicted segments and computes a matching between them based on the euclidean distance and the orientation difference.
        This function assumes that the predictions are sorted in descending order according to their confidence scores.

        Parameters
        ----------
        pred_segments : list
            list of predicted segments [seg1, seg2, ..., segN]. Each segment is a list of points. 
            Example: seg1 = [[u1, v1], [u2, v2], [u3, v3]]; seg2 = [[u1, v1], [u2, v2]]

        gt_points : list
            list of sampled GT points. Each point is represented as [u, v]

        pred_orient : list
            list of orientations of the predicted segments.

        gt_orient : list
            list of orientations of the line segments the GT points are associated with.


        dist_thres : float
            Distance threshold for the matching. 
            A point can only be matched to a segment, if the shortest distance between the point and the segment is below this threshold.

        orient_thres : float
            Orientation threshold for the matching. 
            A point can only be matched to a segment, if the orientation difference between the segment the gt-point belongs to
            and the predicted segment is below this threshold.

        pred_confidence : list, optional
            List of confidence scores for the predicted segments. required if use_confidence is True.

        Returns
        -------
        match_pred_ind : list
            List of indices of the matched predicted points.

        match_gt_ind : list
            List of indices of the matched GT points.

        avg_matching_distance : float
            Average matching distance of the matched points. 
            If no points are matched, returns -1.

        graphs : dict
            If self.extended_summary is False, returns an empty dict.
            keys: confidence scores
            values: list of all edges in the graph as tuples (pred_index, gt_index, cost, orient, is_match).
                Each tuple indicates that there is an edge between the predicted segment at pred_index and the GT point at gt_index with the given cost and orientation difference.
                If no edges are found for a confidence score, the value is an empty list.
        """
        assert all(pred_confidence[i] >= pred_confidence[i+1] for i in range(len(pred_confidence)-1)), "pred_confidence must be sorted in descending order."

        # transform to shapely objects
        pred_segments_shapely = []
        for seg in pred_segments:
            line = LineString(seg)
            pred_segments_shapely.append(line)

        gt_points_shapely = []
        for pt in gt_points:
            point = Point(pt)
            gt_points_shapely.append(point)

        # compute shortest distances between GT points and predicted segments
        euclidean_distances = np.full((len(pred_segments_shapely), len(gt_points_shapely)), np.inf)
        for i, l in enumerate(pred_segments_shapely):
            for j, p in enumerate(gt_points_shapely):
                euclidean_distances[i, j] = l.distance(p)

        # compute angular distances of GT segments and predicted segments
        orient_differences = np.full((len(pred_orient), len(gt_orient)), np.inf)
        gt_orient = np.array(gt_orient)
        for idx_pred, o in enumerate(pred_orient):
            a = np.abs(gt_orient - o)
            b = np.abs(np.abs(gt_orient - o) - 360)
            orient_differences[idx_pred] = np.min(np.stack((a, b)), axis=0)

        matches, graphs = self.confidence_matching( cost_matrix = euclidean_distances.copy(),
                                                    orient_matrix = orient_differences,
                                                    pred_confidence = pred_confidence,
                                                    dist_thres = dist_thres,
                                                    orient_thres = orient_thres)
        
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


    
    def confidence_matching(self, cost_matrix, orient_matrix, pred_confidence, dist_thres=10, orient_thres=10):
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
                                                         orient_thres=orient_thres)
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



    def graph_matching(self, cost_matrix, orient_matrix, dist_thres=10, orient_thres=10):
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

        num_rows, num_cols = cost_matrix.shape
        G = nx.Graph()

        graph_summary = []

        plotting = False

        if self.matching_strategy == 'maximum_matching' or plotting:
            left_set = set()
            # right_set = set()

        # row_appeard = defaultdict(int)
        # col_appeard = defaultdict(int)

        # Add edges only for valid pairs (cost below threshold)
        for i in range(num_rows):
            for j in range(num_cols):
                if cost_matrix[i, j] <= dist_thres and orient_matrix[i, j] <= orient_thres:
                    G.add_edge(f"row_{i}", f"col_{j}", weight=cost_matrix[i, j])  
                    if self.extended_summary:
                        graph_summary.append((i, j, cost_matrix[i, j], orient_matrix[i, j]))

                    if self.matching_strategy == 'maximum_matching' or plotting:
                        left_set.add(f"row_{i}")
                        # right_set.add(f"col_{j}")

                    # row_appeard[i] += 1
                    # col_appeard[j] += 1

        # for i, count in row_appeard.items():
        #     if count > 1:
        #         print(f"Warning: row_{i} appears in {count} edges.")
        # for j, count in col_appeard.items():
        #     if count > 1:
        #         print(f"Warning: col_{j} appears in {count} edges.")

        if plotting:  # visualize graph
            import matplotlib.pyplot as plt

            nx.draw(G, with_labels=True, pos=nx.bipartite_layout(G, left_set))

            pos = nx.spring_layout(G, seed=7)
            nx.draw_networkx_nodes(G, pos, node_size=300)
            edges = [(u, v) for (u, v, d) in G.edges(data=True)]
            nx.draw_networkx_edges(G, pos, edgelist=edges, width=6)
            nx.draw_networkx_labels(G, pos, font_size=20, font_family="sans-serif")
            #edge_labels = nx.get_edge_attributes(G, "weight")
            #nx.draw_networkx_edge_labels(G, pos, edge_labels)

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


    @staticmethod
    def calculate_ap_every_point(rec, prec):
        """
        This function is from https://github.com/rafaelpadilla/review_object_detection_metrics/blob/main/src/evaluators/pascal_voc_evaluator.py
        """
        mrec = []
        mrec.append(0)
        [mrec.append(e) for e in rec]
        mrec.append(1)
        mpre = []
        mpre.append(0)
        [mpre.append(e) for e in prec]
        mpre.append(0)
        for i in range(len(mpre) - 1, 0, -1):
            mpre[i - 1] = max(mpre[i - 1], mpre[i])
        ii = []
        for i in range(len(mrec) - 1):
            if mrec[1:][i] != mrec[0:-1][i]:
                ii.append(i + 1)
        ap = 0
        for i in ii:
            ap = ap + np.sum((mrec[i] - mrec[i - 1]) * mpre[i])
        return [ap, mpre[0:len(mpre) - 1], mrec[0:len(mpre) - 1], ii]


    def plot_evaluation(self, img_ident, image_dir, save_plot_dir, rel_dist_thres):
        """
        Plots the evaluation results for a single image.

        Parameters
        ----------
        img_ident : str
            Identifier of the image to plot (must be a key in self.results[dist_thres]['images']).
        image_dir : str
            Directory where the images are stored.
        save_plot_dir : str
            Directory where the plots will be saved.
        rel_dist_thres : float
            Relative distance threshold for which the evaluation was performed (must be a key in self.results).
        """
        result_key = f"rel_dist_thres_{rel_dist_thres}"

        assert result_key in self.results, f"Relative distance threshold {rel_dist_thres} not found in results."
        assert 'images' in self.results[result_key], "Extended summary was not enabled during evaluation."
        assert img_ident in self.results[result_key]['images'], f"Image {img_ident} not found in results."
        save_plot_dir = os.path.join(save_plot_dir, result_key)

        if not os.path.exists(save_plot_dir):
            os.makedirs(save_plot_dir)

        eval_data = self.results[result_key]['images'][img_ident]

        fig = visualize_segmentwise_evaluation(
            image=os.path.join(image_dir, img_ident),
            pred_segments=eval_data['pred_segments'],
            gt_segments=eval_data['gt_segments'],
            gt_points=eval_data['gt_points'],
            add_gt_rails=False,
            true_positives=eval_data['true_positives'],
            matched_gt_points=eval_data['matched_gt_points'],
            figsize=(15, 8), fontsize=12, dot_size=30, line_thickness=2,
            title=f"Matchings for {img_ident} with rel_dist_thres={rel_dist_thres}"
        )

        fig.axes[0].axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(save_plot_dir, img_ident.split('/')[-1]))
        plt.close(fig)


