```python
{
<img01.png>: {
    'rails': [<polyline1>,
              <polyline2>,
              ...],
    'score': [<confidence1>,
              <confidence2>,
              ... ]
    },
<img02.png>: {...},
...
}

# Each polyline is a list of points: [[u1, v1], [u2, v2], ..., [uN, vN]]  
# A confidence score is a value between 0 and 1
```