import argparse
import os

import numpy as np
import pandas as pd


parser = argparse.ArgumentParser()
parser.add_argument('--dataset_path', type=str, default='./data/CelebA')
parser.add_argument('--k_folds', type=int, default=10)
parser.add_argument('--fold_val_proportion', type=float, default=.5)
args = parser.parse_args()

dataset_path = args.dataset_path
k_folds = args.k_folds
fold_val_proportion = args.fold_val_proportion

train_output_file = os.path.join(dataset_path, f'train_indices.npy')
val_output_file = os.path.join(dataset_path, f'val_indices.npy')

if os.path.exists(train_output_file) or os.path.exists(val_output_file):
    print('Fold splits seem to have already been generated. Delete them manually if you wish to generate them again.')
    exit(0)

if os.path.exists(os.path.join(dataset_path, f'metadata.feather')):
    meta_df = pd.read_feather(os.path.join(dataset_path, f'metadata.feather'))
else:
    print('Metadata has not been computed yet for this dataset. Please run `compute_embeddings.py` first.')
    exit(1)

np.random.seed(42)
train_indices = []
val_indices = []
indices = np.arange(len(meta_df))
num_train_samples = int(len(indices) * (1 - fold_val_proportion))

for _ in range(k_folds):
    np.random.shuffle(indices)
    train_indices.append(indices[:num_train_samples].astype(np.uint32))
    val_indices.append(indices[num_train_samples:].astype(np.uint32))

train_indices = np.stack(train_indices)
val_indices = np.stack(val_indices)
np.save(train_output_file, train_indices)
np.save(val_output_file, val_indices)
