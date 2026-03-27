```python
{
  "images": [
    {"id": 1, "file_name": <name1.png>, "width": <width>, "height": <height>},
    {"id": 2, "file_name": <name2.png>, "width": <width>, "height": <height>},
    ...
  ],
  "categories": [
    {"id": 1, "name": "train"},
    {"id": 2, "name": "catenary_pole"},
    ...
  ],
  "annotations": [
    {"id": <annotation_id>,
     "image_id": <image_id>,
     "category_id": <category_id>,
     "bbox": [u_min, v_min, w, h],
     "occlusion": <occlusion_level>,
     "iscrowd": <boolean>,
     "ignore": <boolean>},
     ...
  ]
}

# booleans are given as 0 (False) or 1 (True)
# occlusion levels: 0 (0-24 %), 1 (25-49 %), 2 (50-74 %), 3 (75-99 %)
```
