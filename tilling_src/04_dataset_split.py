"""
04 - Dataset Splitting (Per-window environments)
Splits window-based environments into train/val/test using stratification on
fire size (detections) and duration.
"""

import sys
from pathlib import Path as _Path
sys.path.append(str((_Path(__file__).parent.parent / 'src').resolve()))

import numpy as np
import pandas as pd
import json
from sklearn.model_selection import StratifiedShuffleSplit

script_dir = _Path(__file__).parent
root_dir = script_dir.parent
embedded_dir = root_dir / 'embedded_data'
tilling_dir = root_dir / 'tilling_data'

print("=" * 80)
print("DATASET SPLITTING (PER-WINDOW)")
print("=" * 80)

# ---------------------------------------------------------------------------
# 1. LOAD MANIFEST + WINDOW META
# ---------------------------------------------------------------------------
print("\n[1/4] Loading environment manifest...")
manifest = pd.read_parquet(tilling_dir / 'environments' / 'environment_manifest.parquet')
windows = pd.read_parquet(embedded_dir / 'sliding_windows_index.parquet')
print(f"  Environments: {len(manifest)}")
print(f"  Windows meta: {len(windows)}")

# Merge
envs = manifest.merge(windows[['window_id','num_detections','duration_hours']], on='window_id', how='left')

# ---------------------------------------------------------------------------
# 2. BUILD STRATIFICATION KEY
# ---------------------------------------------------------------------------
print("\n[2/4] Building stratification key...")

envs['size_cat'] = pd.cut(envs['num_detections'], bins=[0,20,100, np.inf], labels=['small','medium','large'])
envs['dur_cat'] = pd.cut(envs['duration_hours'], bins=[0,24,168, np.inf], labels=['short','medium','long'])

envs['strat_key'] = envs['size_cat'].astype(str) + '_' + envs['dur_cat'].astype(str)
print(f"  Unique strata: {envs['strat_key'].nunique()}")

# ---------------------------------------------------------------------------
# 3. SPLIT
# ---------------------------------------------------------------------------
print("\n[3/4] Performing stratified split...")
strat_counts = envs['strat_key'].value_counts()
min_samples = strat_counts.min()

if min_samples < 3:
    print("  Not enough samples per stratum -> random split fallback")
    idx = np.random.permutation(len(envs))
    n_train = int(0.7*len(envs)); n_val = int(0.15*len(envs))
    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train+n_val]
    test_idx = idx[n_train+n_val:]
else:
    splitter1 = StratifiedShuffleSplit(n_splits=1, test_size=0.30, random_state=42)
    train_idx, temp_idx = next(splitter1.split(envs, envs['strat_key']))
    temp = envs.iloc[temp_idx]
    splitter2 = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
    val_sub_idx, test_sub_idx = next(splitter2.split(temp, temp['strat_key']))
    val_idx = temp_idx[val_sub_idx]
    test_idx = temp_idx[test_sub_idx]

train_envs = envs.iloc[train_idx]['env_id'].tolist()
val_envs = envs.iloc[val_idx]['env_id'].tolist()
test_envs = envs.iloc[test_idx]['env_id'].tolist()

print(f"  Train: {len(train_envs)} Val: {len(val_envs)} Test: {len(test_envs)}")

# ---------------------------------------------------------------------------
# 4. SAVE SPLITS + SUMMARY
# ---------------------------------------------------------------------------
print("\n[4/4] Saving splits...")
splits_meta = {
    'total_envs': len(envs),
    'train': len(train_envs),
    'val': len(val_envs),
    'test': len(test_envs),
    'method': 'stratified' if min_samples >= 3 else 'random',
    'strata': envs['strat_key'].unique().tolist()
}

out_dir = tilling_dir / 'environments'
with open(out_dir / 'train_split.json','w') as f: json.dump(train_envs, f, indent=2)
with open(out_dir / 'val_split.json','w') as f: json.dump(val_envs, f, indent=2)
with open(out_dir / 'test_split.json','w') as f: json.dump(test_envs, f, indent=2)
with open(out_dir / 'splits_metadata.json','w') as f: json.dump(splits_meta, f, indent=2)

# Add split column to manifest and save
manifest['split'] = 'train'
manifest.loc[manifest['env_id'].isin(val_envs), 'split'] = 'val'
manifest.loc[manifest['env_id'].isin(test_envs), 'split'] = 'test'
manifest.to_parquet(tilling_dir / 'environment_manifest.parquet', index=False)
print(f"  Updated manifest with split column")

print("\nSummary:")
print(json.dumps(splits_meta, indent=2))
print("\n" + "=" * 80)
print("DATASET SPLITTING COMPLETE")
print("=" * 80)
