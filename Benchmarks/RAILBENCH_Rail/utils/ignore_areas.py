import numpy as np
from shapely import geometry


def validate_polylines(data, data_type="predictions"):
    """
    Check for degenerate polylines (fewer than 2 points) that would cause
    errors in downstream metric computation (e.g. ChamferAP, LineAP).

    Parameters
    ----------
    data : dict
        If data_type == "predictions":
            {image_name: {'rails': [polyline, ...], 'score': [float, ...]}}
        If data_type == "gt":
            Ground truth in RailBench/COCO format with 'images', 'categories',
            and 'annotations' keys.

    data_type : str
        Either "predictions" or "gt".

    Returns
    -------
    issues : list[dict]
        List of dicts with keys 'image', 'rail_index', 'num_points', 'polyline'
        for every degenerate polyline found. Empty list means no issues.
    """
    issues = []

    if data_type == "gt":
        img_id_name = {img["id"]: img["file_name"] for img in data["images"]}
        cat_id_name = {cat["id"]: cat["name"] for cat in data["categories"]}

        for ann in data["annotations"]:
            cat_name = cat_id_name.get(ann["category_id"], "")
            if cat_name != "rail":
                continue
            polyline = ann.get("polyline", [])
            n_pts = len(polyline)
            if n_pts < 2:
                issues.append({
                    "image": img_id_name.get(ann["image_id"], "unknown"),
                    "annotation_id": ann["id"],
                    "num_points": n_pts,
                    "polyline": polyline,
                })

    elif data_type == "predictions":
        for img_name, pred in data.items():
            for i, rail in enumerate(pred["rails"]):
                n_pts = len(rail)
                if n_pts < 2:
                    issues.append({
                        "image": img_name,
                        "rail_index": i,
                        "num_points": n_pts,
                        "polyline": rail,
                    })
    else:
        raise ValueError(f"data_type must be 'predictions' or 'gt', got '{data_type}'")

    if issues:
        print(f"Found {len(issues)} degenerate polyline(s) in {data_type}:")
        for issue in issues:
            print(f"  {issue}")
    else:
        print(f"No degenerate polylines found in {data_type}.")

    return issues


def process_predictions(preds, gt):
    """
    Remove parts of predictions that fall within the specified ignore areas.
    """

    # step 1: get ignore areas for each img
    for cat in gt['categories']:
        if cat['name'] == 'ignore_area':
            cat_id_ignore = cat['id']
            break

    ignore_areas = dict()
    img_id_mapping = {img['file_name']: img['id'] for img in gt['images']}

    for img in preds.keys():
        img_id = img_id_mapping[img]
        ignore_areas[img] = []

        for ann in gt['annotations']:
            if ann['image_id'] == img_id and ann['category_id'] == cat_id_ignore:
                ignore_areas[img].append(ann['polygon'])

    # step 2: iterate over each image and adapt predictions
    preds_new = dict()

    for img, pred in preds.items():
        preds_new[img] = {'rails': [], 'score': []}
        ignore_polgons = [geometry.Polygon(area) for area in ignore_areas[img]] # all ignore areas for the specific image

        for i, rail in enumerate(pred['rails']):
            polyline = geometry.LineString(rail) 
            lines_new = cut_polyline(polyline, ignore_polgons)
            for line in lines_new:
                preds_new[img]['rails'].append(line)
                preds_new[img]['score'].append(pred['score'][i])

    return preds_new


def cut_polyline(polyline, ignore_polygons, length_threshold=20):
    lines_new = []
    # subtract ignore areas from the polyline
    remaining = polyline
    for ia in ignore_polygons:
        if remaining.is_empty:
            break
        if isinstance(remaining, geometry.LineString):
            if remaining.intersects(ia):
                remaining = remaining.difference(ia)
        elif isinstance(remaining, geometry.MultiLineString):
            new_remaining = []
            for geom in remaining.geoms:
                if geom.intersects(ia):
                    diff = geom.difference(ia)
                    if isinstance(diff, geometry.LineString):
                        new_remaining.append(diff)
                    elif isinstance(diff, geometry.MultiLineString):
                        new_remaining.extend(diff.geoms)
                else:
                    new_remaining.append(geom)
            remaining = geometry.MultiLineString(new_remaining)
        

    if isinstance(remaining, geometry.LineString):
        if not remaining.is_empty and len(remaining.coords) >= 2 and remaining.length > length_threshold:
            points_new = [list(pt) for pt in remaining.coords]
            lines_new.append(points_new)
    elif isinstance(remaining, geometry.MultiLineString):
        for geom in remaining.geoms:
            if geom.length > length_threshold and len(geom.coords) >= 2:
                points_new = [list(pt) for pt in geom.coords]
                lines_new.append(points_new)
    else:
        print(type(remaining))
        print(f"Remaining is of type {type(remaining)}; line is skipped.")

    return lines_new