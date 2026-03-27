```python

{
  "images": [
    {"id": 1, "file_name": <img01.png>, "width": <width>, "height": <height>},
    {"id": 2, "file_name": <img02.png>, "width": <width>, "height": <height>},
    ...
  ],
  "categories": [
    {"id": 1, "name": "rail"},
    {"id": 2, "name": "ignore_area"}
  ],
  "annotations": [
    {"id": <annotation_id>, # unqiue for each annotation
     "image_id": <image_id>, # corresponds to the "ids" in the "images" list 
     "category_id": <category_id>, # corresponds to the "ids" in the "categories" list
     "polyline": [[u1, v1], [u2, v2], ...], # if rail 
     "polygon": [[u1, v1], [u2, v2], ...], # if ignore area
     "occlusion": <occlusion_level>  # if rail
     "rightRail": <boolean> # if rail
     }
  ]
}

# "rightRail" is either 1 for right rail or 0 for left rail
# occlusion levels: 0 (0-24 %), 1 (25-49 %), 2 (50-74 %), 3 (75-99 %)

```
