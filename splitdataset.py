# =====================================================
# SPLIT DATASET: 5 Sleeping + 15 Falling → TEST (not use 80:20 ratio as usual since there are so little sleeping,
# -> if use 20% test --> just 2 sleeping for testing
#                Rest → TRAIN (for augmentation)
# =====================================================

import pandas as pd
from sklearn.model_selection import train_test_split
import os

# ================== CONFIG ==================
full_csv = "/media/public_data/temp/Phuong/fall_videos/GMDCSA24/LSTM/GMDCSA24_yolo11pose_All_10fps_Filter.csv"


train_csv_output = "/media/public_data/temp/Phuong/fall_videos/GMDCSA24/LSTM/GMDCSA24_10fps_train_raw.csv"
test_csv_output  = "/media/public_data/temp/Phuong/fall_videos/GMDCSA24/LSTM/GMDCSA24_10fps_test_raw.csv"

# Number to reserve for test
N_SLEEPING_TEST = 5
N_FALLING_TEST  = 15

# Seed for reproducibility (same split every time)
RANDOM_SEED = 42
# ===========================================

print("Loading full dataset...")
df = pd.read_csv(full_csv)

# Check how many
n_sleeping = len(df[df['class'] == 'Sleeping']['video_path'].unique())
n_falling  = len(df[df['class'] == 'Falling']['video_path'].unique())
print(f"Total real sequences → Sleeping: {n_sleeping} | Falling: {n_falling}")

if n_sleeping < N_SLEEPING_TEST:
    raise ValueError(f"You only have {n_sleeping} sleeping videos, but want {N_SLEEPING_TEST} for test!")
if n_falling < N_FALLING_TEST:
    raise ValueError(f"You only have {n_falling} falling videos, but want {N_FALLING_TEST} for test!")

# Get unique video paths per class
sleeping_paths = df[df['class'] == 'Sleeping']['video_path'].unique()
falling_paths  = df[df['class'] == 'Falling']['video_path'].unique()

# Split each class separately
sleeping_test  = pd.Series(sleeping_paths).sample(n=N_SLEEPING_TEST, random_state=RANDOM_SEED).tolist()
falling_test   = pd.Series(falling_paths).sample(n=N_FALLING_TEST, random_state=RANDOM_SEED).tolist()

test_paths = sleeping_test + falling_test

# Create train and test dataframes
test_df  = df[df['video_path'].isin(test_paths)].copy()
train_df = df[~df['video_path'].isin(test_paths)].copy()

# Double-check
print("\nFINAL SPLIT:")
print(f"→ Test set  : Sleeping = {len(test_df[test_df['class']=='Sleeping']['video_path'].unique())} | "
      f"Falling = {len(test_df[test_df['class']=='Falling']['video_path'].unique())}")
print(f"→ Train set : Sleeping = {len(train_df[train_df['class']=='Sleeping']['video_path'].unique())} | "
      f"Falling = {len(train_df[train_df['class']=='Falling']['video_path'].unique())}")

# Save
train_df.to_csv(train_csv_output, index=False)
test_df.to_csv(test_csv_output, index=False)

print("\nSAVED:")
print(f"   Train (for augmentation) → {train_csv_output}")
print(f"   Test  (clean, real)      → {test_csv_output}")
