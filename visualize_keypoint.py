#!/usr/bin/env python3
"""
Visualize MediaPipe pose keypoints on original images.
Compatible with Python 3.12+
"""

import cv2
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import argparse

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------
BASE_DIR   = Path("/media/public_data/temp/Phuong/fall_videos/Sort")
FALL_DIR   = BASE_DIR / "fall"
NON_FALL_DIR = BASE_DIR / "non_fall"
TXT_DIR    = BASE_DIR / "pose_keypoints_txt"

# MediaPipe connections
MP_POSE_CONNECTIONS = [
    (0,1), (1,2), (2,3), (3,7), (0,4), (4,5), (5,6), (6,8),
    (9,10),
    (11,13), (13,15), (15,17), (15,19), (15,21),
    (11,12), (12,14), (14,16), (16,18), (16,20), (16,22),
    (11,23), (12,24), (23,24),
    (23,25), (25,27), (27,29), (27,31),
    (24,26), (26,28), (28,30), (28,32)
]

COLOR_SKELETON = (0, 255, 0)
COLOR_LANDMARK = (255, 0, 0)


def draw_pose(img_path: Path, kp_df: pd.DataFrame, out_path: Path) -> None:
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"Warning: Cannot read {img_path}")
        return

    for _, row in kp_df.iterrows():
        x_px, y_px = int(row["x_px"]), int(row["y_px"])
        cv2.circle(img, (x_px, y_px), 5, COLOR_LANDMARK, -1)

    for a, b in MP_POSE_CONNECTIONS:
        pt_a = (int(kp_df.iloc[a]["x_px"]), int(kp_df.iloc[a]["y_px"]))
        pt_b = (int(kp_df.iloc[b]["x_px"]), int(kp_df.iloc[b]["y_px"]))
        cv2.line(img, pt_a, pt_b, COLOR_SKELETON, 2)

    cv2.imwrite(str(out_path), img)


def process_one(txt_file: Path, img_name: str) -> None:
    df = pd.read_csv(txt_file, sep="\t", header=None,
                     names=["image","landmark","x","y","z","vis","x_px","y_px"])
    kp = df[df["image"] == img_name]
    if kp.empty:
        print(f"No keypoints for {img_name}")
        return

    img_path = FALL_DIR / img_name
    if not img_path.exists():
        img_path = NON_FALL_DIR / img_name
    if not img_path.exists():
        print(f"Image not found: {img_name}")
        return

    out_path = img_path.with_name(f"{img_path.stem}_annotated.jpg")
    draw_pose(img_path, kp, out_path)
    print(f"Saved → {out_path}")


def process_all(txt_file: Path) -> None:
    df = pd.read_csv(txt_file, sep="\t", header=None,
                     names=["image","landmark","x","y","z","vis","x_px","y_px"])
    images = df["image"].unique()
    print(f"Found {len(images)} frames in {txt_file.name}")

    for img_name in tqdm(images, desc="Annotating"):
        kp = df[df["image"] == img_name]
        img_path = FALL_DIR / img_name
        if not img_path.exists():
            img_path = NON_FALL_DIR / img_name
        if not img_path.exists():
            continue
        out_path = img_path.with_name(f"{img_path.stem}_annotated.jpg")
        draw_pose(img_path, kp, out_path)


# ----------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize MediaPipe pose keypoints")
    parser.add_argument("--all", action="store_true", help="Annotate all images")
    parser.add_argument("--image", type=str, help="Single image name, e.g. frame_00011_fall.jpg")
    parser.add_argument("--class", choices=["fall", "non_fall"], default="fall",
                        help="Which txt file to read (default: fall)")

    args = parser.parse_args()

    # SAFE: Use .format() instead of f-string
    txt_file = TXT_DIR / "{}_keypoints.txt".format(args.class)

    if not txt_file.exists():
        print(f"Keypoint file not found: {txt_file}")
        raise SystemExit(1)

    if args.all:
        process_all(txt_file)
    elif args.image:
        process_one(txt_file, args.image)
    else:
        parser.print_help()