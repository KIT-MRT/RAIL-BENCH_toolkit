```python

[
    {
        "name": <name1.png>,
        "index": 0,
        "labels": [
            {
                "id": <track_id>,
                "category": "person",
                "bbox": [u_min, v_min, w, h]
            },
            {
                "id": <track_id>,
                "category": "person",
                "bbox": [u_min, v_min, w, h]
            },
            ...
        ]
    },

    {
        "name": <name2.png>,
        "index": 1,
        "labels": [
            ...
        ]
    },

    ...

]

# track_id can be an integer, float or string
```