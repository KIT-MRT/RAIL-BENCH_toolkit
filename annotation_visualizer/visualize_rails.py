import argparse
import sys
import os

# Add the toolkit root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import cv2
import numpy as np

from Benchmarks.RAILBENCH_Rail.viz.viz_lines import visualize_tracks, railbench_preparation
from utils.viz.viz_image import image_preparation
from utils.helpers import load_json


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize RailBench annotations interactively.")
    parser.add_argument("--annotations", "-a", required=True, help="Path to the JSON annotations file.")
    parser.add_argument("--image_dir", "-i", required=True, help="Path to the folder containing images.")
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"Loading annotations from {args.annotations}...")
    anns = load_json(args.annotations)

    images = anns.get("images", [])
    if not images:
        print("No images found in annotations.")
        sys.exit(1)

    n = len(images)
    print(f"Found {n} images.")
    print("Controls: LEFT/RIGHT or A/D — prev/next | UP/DOWN — ±10 | 0-9+Enter — jump to image | F/L — first/last | S — save | H — help | Q/ESC — quit")

    # waitKeyEx codes for arrow keys (Linux X11)
    KEY_LEFT  = 0xFF51
    KEY_RIGHT = 0xFF53
    KEY_UP    = 0xFF52
    KEY_DOWN  = 0xFF54

    HELP_LINES = [
        "Navigation:",
        "  LEFT / A         : previous image",
        "  RIGHT / D        : next image",
        "  UP               : jump back 10",
        "  DOWN             : jump forward 10",
        "  F                : first image",
        "  L                : last image",
        "  0-9 then Enter   : jump to image number",
        "  ESC              : cancel input / quit",
        "Other:",
        "  S                : save current view",
        "  H                : toggle this help",
        "  Q                : quit",
    ]

    idx = 0
    show_help = False
    window_name = "RailBench Annotation Visualizer"
    cv2.namedWindow(window_name, cv2.WINDOW_GUI_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)

    while True:
        image_info = images[idx]
        image_id = image_info["id"]
        file_name = image_info["file_name"]
        image_path = os.path.join(args.image_dir, file_name)

        print(f"\r[{idx + 1}/{n}] {file_name}", end="", flush=True)

        try:
            image = image_preparation(image_path)
            rails, ignore_areas = railbench_preparation(anns, image_id=image_id)
            img_viz = visualize_tracks(image, rails, ignore_areas, instance_coloring=True)
            img_bgr = cv2.cvtColor(np.array(img_viz), cv2.COLOR_RGB2BGR)
        except Exception as e:
            print(f"\nError processing {file_name}: {e}")
            img_bgr = np.zeros((720, 1280, 3), dtype=np.uint8)
            cv2.putText(img_bgr, f"Error: {e}", (30, 360),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        num_buffer = ""
        status_msg = ""

        def render():
            d = img_bgr.copy()
            h, w = d.shape[:2]
            font = cv2.FONT_HERSHEY_SIMPLEX
            bar_fs = 1.0   # font scale for top/bottom bars
            bar_h  = 50    # height of each bar

            # Top bar: semi-transparent black strip
            overlay = d.copy()
            cv2.rectangle(overlay, (0, 0), (w, bar_h), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.55, d, 0.45, 0, d)

            # "[H] for help" hint (top-left)
            hint = "[H] for help"
            cv2.putText(d, hint, (12, bar_h - 12),
                        font, bar_fs, (180, 180, 180), 2, cv2.LINE_AA)

            # Bottom bar: semi-transparent black strip
            overlay = d.copy()
            cv2.rectangle(overlay, (0, h - bar_h), (w, h), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.55, d, 0.45, 0, d)

            # Image name + index (bottom-left)
            status = f"[{idx + 1}/{n}]  {file_name}"
            cv2.putText(d, status, (12, h - 14),
                        font, bar_fs, (230, 230, 230), 2, cv2.LINE_AA)

            # Help overlay (anchored below top bar so it doesn't overlap)
            if show_help:
                font_scale = 1.2
                line_gap = 52
                pad_x, pad_y = 24, 18
                text_x = 24
                box_y1 = bar_h + 8
                text_y0 = box_y1 + pad_y + line_gap

                box_w = max(
                    cv2.getTextSize(line, font, font_scale, 2)[0][0]
                    for line in HELP_LINES
                ) + 2 * pad_x + text_x
                box_h_help = pad_y + len(HELP_LINES) * line_gap + pad_y
                box_x1 = 10
                box_x2 = box_x1 + box_w
                box_y2 = box_y1 + box_h_help

                overlay = d.copy()
                cv2.rectangle(overlay, (box_x1, box_y1), (box_x2, box_y2), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.65, d, 0.35, 0, d)
                for i, line in enumerate(HELP_LINES):
                    cv2.putText(d, line, (text_x, text_y0 + i * line_gap),
                                font, font_scale, (220, 220, 220), 2, cv2.LINE_AA)

            # Number-jump input prompt
            if num_buffer:
                prompt = f"Go to image: {num_buffer}_  (Enter=jump, ESC=cancel)"
                cv2.putText(d, prompt, (15, h - bar_h - 14),
                            font, bar_fs, (0, 255, 255), 2, cv2.LINE_AA)

            # Transient status message (e.g. "Saved: ...")
            if status_msg:
                cv2.putText(d, status_msg, (15, h - bar_h - 14),
                            font, bar_fs, (100, 255, 100), 2, cv2.LINE_AA)

            cv2.imshow(window_name, d)

        render()

        while True:
            key = cv2.waitKeyEx(0)

            if ord('0') <= key <= ord('9'):
                num_buffer += chr(key)
                status_msg = ""
                render()

            elif key in (8, 0xFF08):  # Backspace
                num_buffer = num_buffer[:-1]
                render()

            elif key in (13, 0xFF0D):  # Enter — confirm jump
                if num_buffer:
                    target = int(num_buffer) - 1
                    idx = max(0, min(n - 1, target))
                    num_buffer = ""
                    break

            elif key == 27:  # ESC — cancel input or quit
                if num_buffer:
                    num_buffer = ""
                    render()
                else:
                    print("\nExiting.")
                    cv2.destroyAllWindows()
                    sys.exit(0)

            elif key in (KEY_LEFT, ord('a'), ord('A')):
                idx = (idx - 1) % n
                break
            elif key in (KEY_RIGHT, ord('d'), ord('D')):
                idx = (idx + 1) % n
                break
            elif key == KEY_UP:
                idx = (idx - 10) % n
                break
            elif key == KEY_DOWN:
                idx = (idx + 10) % n
                break

            elif key in (ord('f'), ord('F')):
                idx = 0
                break
            elif key in (ord('l'), ord('L')):
                idx = n - 1
                break

            elif key in (ord('s'), ord('S')):
                save_dir = "annotation_visualizer/saved_frames/rails"
                os.makedirs(save_dir, exist_ok=True)
                base = os.path.splitext(os.path.basename(file_name))[0]
                save_path = os.path.join(save_dir, f"{base}_annotated.png")
                cv2.imwrite(save_path, img_bgr)
                status_msg = f"Saved: {save_path}"
                print(f"\nSaved: {save_path}")
                render()

            elif key in (ord('h'), ord('H')):
                show_help = not show_help
                render()

            elif key in (ord('q'), ord('Q')):
                print("\nExiting.")
                cv2.destroyAllWindows()
                sys.exit(0)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()