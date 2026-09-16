    
def polyline_orientation(polyline: list) -> list:
    """Ensure polyline is oriented foreground → background (large v → small v)."""
    if len(polyline) < 2:
        return polyline
    if polyline[0][1] < polyline[-1][1]:  # start_v < end_v → flip
        return polyline[::-1]
    return polyline