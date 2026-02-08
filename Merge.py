
# After covert_video_to CSV, using YOLOv11-pose for each subject, now merge them into 1 file

import pandas as pd
import os
import glob

# -------------------------- CONFIG --------------------------

BASE_PATH = "/media/public_data/temp/Phuong/fall_videos/GMDCSA24/LSTM"

# Automatically find all Subject CSV files
csv_files = glob.glob(os.path.join(BASE_PATH, "**", "GMDCSA24_yolo11pose_S*_10fps.csv"), recursive=True)

# Sort them : S1, S2, S3, S4
csv_files = sorted(csv_files)
print("Found these files:")
for f in csv_files:
    print("  →", os.path.basename(f))

# -------------------------- Merge --------------------------
dfs = []
for file in csv_files:
    print(f"Loading {os.path.basename(file)} ...")
    df = pd.read_csv(file)
    print(f"   → {len(df):,} frames, {df['video_path'].nunique()} videos, classes: {sorted(df['class'].unique())}")
    dfs.append(df)

print("\nMerging all subjects...")
full_df = pd.concat(dfs, ignore_index=True)

# -------------------------- Final check & save --------------------------
output_file = os.path.join(BASE_PATH, "GMDCSA24_yolo11pose_All_10fps.csv")
full_df.to_csv(output_file, index=False)

print("\n" + "="*80)
print("All 4 subjects merged into one file")
print(f"   Output → {output_file}")
print(f"   Total frames   → {len(full_df):,}")
print(f"   Total videos   → {full_df['video_path'].nunique()}")
print(f"   Total subjects → {full_df['video_path'].apply(lambda x: x.split('/')[0] if isinstance(x,str) else '').nunique()}")
print(f"   Classes → {sorted(full_df['class'].unique())}")
print("="*80)