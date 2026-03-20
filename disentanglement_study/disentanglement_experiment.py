import argparse
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from tqdm import tqdm

warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
warnings.filterwarnings('ignore', category=FutureWarning, module='sklearn')

# --- 1. Arguments & Configuration ---
parser = argparse.ArgumentParser(description="Run disentanglement experiment (leakage test) on precomputed features.")
parser.add_argument('--main_task', required=True, help="Main task attribute (e.g., 'profession')")
parser.add_argument('--bias_attr', required=True, help="Bias attribute (e.g., 'gender' or 'ethnicity')")
parser.add_argument('--clip_model', required=True, help="CLIP model name (e.g., 'ViT-B/16')")
parser.add_argument('--latent', required=True, choices=['sae', 'clip'], help="Feature type to use ('sae' or 'clip')")
args = parser.parse_args()

MAIN_TASK = args.main_task
BIAS_ATTR = args.bias_attr
CLIP_MODEL_NAME = args.clip_model
FEATURE_TYPE = args.latent

# --- Define Tasks ---
MAIN_TARGET_COLUMN = MAIN_TASK
SECOND_TARGET_COLUMN = BIAS_ATTR

# --- Experiment Parameters ---
N_SPLITS_KFOLD = 5
RANDOM_STATE = 42

# --- Dynamic Filename Generation ---
SAFE_MODEL_NAME = CLIP_MODEL_NAME.replace("/", "_").replace("@", "_")

# Matches the filename generation in the precomputation script
LABEL_FILE = f"clip_labels_{BIAS_ATTR}_{MAIN_TASK}_{SAFE_MODEL_NAME}.csv"

if FEATURE_TYPE == 'sae':
    FEATURE_FILE = f"sae_latents_{BIAS_ATTR}_{MAIN_TASK}_{SAFE_MODEL_NAME}.npy"
    FEATURE_NAME_PRINT = "SAE Activations"
else:
    FEATURE_FILE = f"clip_embeddings_{BIAS_ATTR}_{MAIN_TASK}_{SAFE_MODEL_NAME}.npy"
    FEATURE_NAME_PRINT = "CLIP Embeddings"

# --- 2. Load Precomputed Data ---

print("--- Configuration ---")
print(f"  Main Task:   {MAIN_TARGET_COLUMN}")
print(f"  Second Task: {SECOND_TARGET_COLUMN}")
print(f"  CLIP Model:  {CLIP_MODEL_NAME}")
print(f"  Features:    {FEATURE_NAME_PRINT}")
print("---------------------\n")

try:
    print(f"Loading features from: {FEATURE_FILE}")
    X = np.load(FEATURE_FILE) 
    
    print(f"Loading labels from: {LABEL_FILE}")
    df_labels = pd.read_csv(LABEL_FILE)
except FileNotFoundError as e:
    print(f"\nError: Could not find file '{e.filename}'")
    print("Please make sure your configuration matches the precomputed filenames.")
    sys.exit(1)

y_main = df_labels[MAIN_TARGET_COLUMN]
y_second = df_labels[SECOND_TARGET_COLUMN]

print(f"\nFeature matrix (X) shape: {X.shape}")
print(f"Main target '{MAIN_TARGET_COLUMN}' (y_main) shape: {y_main.shape}")
print(f"Second target '{SECOND_TARGET_COLUMN}' (y_second) shape: {y_second.shape}")


# --- 3. Run Full Combined Experiment with Cross-Validation ---

print("\n" + "="*50)
print(f"🔄 Running {N_SPLITS_KFOLD}-Fold Combined Cross-Validation")
print("="*50)

# Stratify on the main, high-cardinality task
kfold = StratifiedKFold(n_splits=N_SPLITS_KFOLD, shuffle=True, random_state=RANDOM_STATE)

scores_direct_probe_main = []   
scores_direct_probe_second = []  
scores_seq_second_from_main = [] 

for fold, (train_index, test_index) in enumerate(tqdm(kfold.split(X, y_main), total=N_SPLITS_KFOLD, desc="Cross-Validation Folds")):
    
    X_train, X_test = X[train_index], X[test_index]
    y_main_train, y_main_test = y_main.values[train_index], y_main.values[test_index]
    y_second_train, y_second_test = y_second.values[train_index], y_second.values[test_index]

    # === B. Train & Evaluate Direct Probes ===
    
    probe_Main_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', LogisticRegression(max_iter=1000, random_state=RANDOM_STATE))
    ])
    probe_Main_pipeline.fit(X_train, y_main_train)
    
    probe_Second_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', LogisticRegression(max_iter=1000, random_state=RANDOM_STATE))
    ])
    probe_Second_pipeline.fit(X_train, y_second_train)

    pred_main = probe_Main_pipeline.predict(X_test)
    scores_direct_probe_main.append(accuracy_score(y_main_test, pred_main))
    
    pred_direct_Second = probe_Second_pipeline.predict(X_test)
    scores_direct_probe_second.append(accuracy_score(y_second_test, pred_direct_Second))

    # === C. Get Logits from Direct Probes ===

    scaler_Main = probe_Main_pipeline.named_steps['scaler']
    model_Main = probe_Main_pipeline.named_steps['model']
    X_train_scaled_Main = scaler_Main.transform(X_train)
    X_test_scaled_Main = scaler_Main.transform(X_test)
    
    X_logits_Main_train = model_Main.decision_function(X_train_scaled_Main)
    X_logits_Main_test = model_Main.decision_function(X_test_scaled_Main)

    # === D. Train & Evaluate Sequential Probes (Leakage Tests) ===

    probe_Second_on_logits_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', LogisticRegression(max_iter=1000, random_state=RANDOM_STATE))
    ])
    probe_Second_on_logits_pipeline.fit(X_logits_Main_train, y_second_train)

    pred_seq_Second = probe_Second_on_logits_pipeline.predict(X_logits_Main_test)
    scores_seq_second_from_main.append(accuracy_score(y_second_test, pred_seq_Second))

# --- 4. Report Final Results ---

num_classes_main = len(y_main.unique())
baseline_main = 1.0 / num_classes_main
num_classes_second = len(y_second.unique())
baseline_second = 1.0 / num_classes_second

def report_scores(title, scores, baseline, baseline_classes):
    print("\n" + "-"*50)
    print(f"📊 {title}")
    print(f"   Average Accuracy: {np.mean(scores):.4f} +/- {np.std(scores):.4f}")
    print(f"   Baseline Accuracy:  {baseline:.4f} (1 / {baseline_classes} classes)")

print("\n" + "="*50)
print("✅ EXPERIMENT RESULTS")
print(f"Features: {FEATURE_NAME_PRINT} ({FEATURE_FILE})")
print(f"Model: {CLIP_MODEL_NAME}")
print("="*50)

report_scores(
    f"Task 1: Predict '{MAIN_TARGET_COLUMN}' from Activations (Direct Probe)",
    scores_direct_probe_main,
    baseline_main,
    num_classes_main
)

report_scores(
    f"Task 2: Predict '{SECOND_TARGET_COLUMN}' from Activations (Direct Probe)",
    scores_direct_probe_second,
    baseline_second,
    num_classes_second
)

report_scores(
    f"Task 3: Predict '{SECOND_TARGET_COLUMN}' from '{MAIN_TARGET_COLUMN}' Logits (Leakage)",
    scores_seq_second_from_main,
    baseline_second,
    num_classes_second
)

print("\n" + "-"*50)
print("--- Experiment Complete ---")
