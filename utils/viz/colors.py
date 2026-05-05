BRIGHT_COLORS_RGB = [
    (255,   0,   0),   # Blue
    (0,   255,   0),   # Green
    (0,     0, 255),   # Red
    (255, 255,   0),   # Cyan
    (255,   0, 255),   # Magenta
    (0,   255, 255),   # Yellow
    (255, 128,   0),   # Orange
    (128,   0, 255),   # Purple
    (0,   128, 255),   # Light Orange
    (128, 255,   0),   # Lime
    (255,   0, 128),   # Pink
    (0,   255, 128),   # Mint
    (128,   0,   0),   # Dark Blue
    (0,   128,   0),   # Dark Green
    (0,     0, 128),   # Dark Red
    (128, 128,   0),   # Olive
    (128,   0, 128),   # Violet
    (0,   128, 128),   # Teal
    (200, 200,   0),   # Bright Yellow
    (200,   0, 200),   # Bright Magenta
    (0,   200, 200),   # Bright Cyan
    (255, 180,   80),  # Peach
    (180, 255,  80),   # Light Green
    (80,  180, 255),   # Sky Blue
    (255,  80, 180),   # Rose
]

GT_COLOR = "#00a2ff"
PRED_COLOR = "#ffae00"
TP_COLOR = "#2eff04"
TP2_COLOR ="#3bc021"
F_COLOR = "#ff0000"
F2_COLOR = "#f700ff"
GREY_COLOR = "#bbbaba"
GT_COLOR_LIGHT = "#a3c9f7"
PRED_COLOR_LIGHT = "#f7d9a3"

def hex_to_rgb(hexstr: str):
    s = hexstr.lstrip("#")
    if len(s) != 6:
        raise ValueError("Expect hex in RRGGBB format")
    # return (R, G, B)
    return tuple(int(s[i:i+2], 16) for i in (0, 2, 4))
