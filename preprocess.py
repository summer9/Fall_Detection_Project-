# Find the distribution of the dataset
# Split the dataset: 80/20 train: test



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os

# -------------------------- CONFIG --------------------------
csv_file = "/media/public_data/temp/Phuong/fall_videos/GMDCSA24/LSTM/GMDCSA24_yolo11pose_All_10fps_Filter.csv"

# Output files
output_dir = os.path.dirname(csv_file)


print("Loading CSV and analyzing sequence lengths by class...\n")
df = pd.read_csv(csv_file)

# Verify pose columns exist
pose_cols = [col for col in df.columns if col.endswith(('_x', '_y', '_c'))]
assert len(pose_cols) == 51, f"Expected 51 pose columns, got {len(pose_cols)}"

# -------------------------- Group by video and assign class --------------------------
print("Grouping frames by video_path and determining dominant class...")
video_info = []

for video_path in df['video_path'].unique():
    seq = df[df['video_path'] == video_path]
    num_frames = len(seq)

    # Use the most frequent class in the video (more robust than first row)
    dominant_class = seq['class'].value_counts().index[0].strip()

    label = 0 if 'leeping' in dominant_class.lower() else 1   # Sleeping = 0, Falling = 1
    class_name = "Sleeping" if label == 0 else "Falling"

    video_info.append({
        'video_path': video_path,
        'num_frames': num_frames,
        'class': class_name,
        'label': label
    })

video_df = pd.DataFrame(video_info)

# -------------------------- Stats & Plot (same as before) --------------------------
sleeping_lengths = video_df[video_df['label'] == 0]['num_frames']
falling_lengths  = video_df[video_df['label'] == 1]['num_frames']

def print_stats(name, data):
    if len(data) == 0:
        print(f"{name}: No videos")
        return
    print(f"{name} Videos: {len(data)}")
    print(f"  Min: {data.min():4d} frames (~{data.min()/10:.1f}s @10fps)")
    print(f"  Max: {data.max():4d} frames (~{data.max()/10:.1f}s)")
    print(f"  Mean: {data.mean():.1f} frames (~{data.mean()/10:.1f}s)")
    print(f"  Median: {np.median(data):.0f} frames\n")

print("=" * 80)
#print("SEQUENCE LENGTH ANALYSIS (10 fps)")
print("=" * 80)
print(f"Total videos: {len(video_df)} | Sleeping: {len(sleeping_lengths)} | Falling: {len(falling_lengths)}\n")
print_stats("SLEEPING", sleeping_lengths)
print_stats("FALLING", falling_lengths)

# Plot histogram
plt.figure(figsize=(12, 6))
plt.hist(sleeping_lengths, bins=50, alpha=0.7, label='Sleeping', color='skyblue', edgecolor='black')
plt.hist(falling_lengths, bins=30, alpha=0.8, label='Falling', color='salmon', edgecolor='black')
plt.axvline(np.median(sleeping_lengths), color='blue', linestyle='--', label=f"Sleeping Median: {np.median(sleeping_lengths):.0f}")
plt.axvline(np.median(falling_lengths), color='red', linestyle='--', label=f"Falling Median: {np.median(falling_lengths):.0f}")
plt.xlabel('Sequence Length (frames @10fps)', fontsize=17)
plt.ylabel('Number of Videos', fontsize=17)
#plt.title('Sequence Length Distribution by Class', fontsize=16)
plt.legend(fontsize=15)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# -------------------------- 80/20 Train-Test Split --------------------------
print("\n" + "=" * 80)
print("PERFORMING 80/20 TRAIN-TEST SPLIT")
print("=" * 80)

train_videos, test_videos = train_test_split(
    video_df,
    test_size=0.20,
    random_state=42,
    stratify=video_df['label']   # This ensures same % of Sleeping/Falling in both sets
)

print(f"Train videos: {len(train_videos)} ({len(train_videos)/len(video_df)*100:.1f}%)")
print(f"Test videos:  {len(test_videos)}  ({len(test_videos)/len(video_df)*100:.1f}%)")
print(f"Train Sleeping/Falling: {train_videos['class'].value_counts().to_dict()}")
print(f"Test  Sleeping/Falling: {test_videos['class'].value_counts().to_dict()}")

# -------------------------- Create train/test DataFrames --------------------------
train_paths = train_videos['video_path'].tolist()
test_paths  = test_videos['video_path'].tolist()

train_df = df[df['video_path'].isin(train_paths)].copy()
test_df  = df[df['video_path'].isin(test_paths)].copy()

# Optional: add split column for debugging
train_df['split'] = 'train'
test_df['split']  = 'test'

