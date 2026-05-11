```python

{
    "video_name": <video_name>,
    "width": <image_width>, 
    "height": <image_height>,
    "annotations": [
        {
            "name": <name1.png>,
            "index": 0,
            "labels": [

                {"id": <track_id>,
                "category": "person",
                "attributes": {
                    "iscrowd": <boolean>
                    },
                "bbox": [u_min, v_min, w, h]},

                {"id": <track_id>,
                "category": "person",
                "attributes": {
                    "iscrowd": <boolean>
                    },
                "bbox": [u_min, v_min, w, h]},

                ...

                ]
        },
        
        {
            "name": <name2.png>,
            "index": 1,
            "labels": [...],
        },
        ...

    ]
}



```