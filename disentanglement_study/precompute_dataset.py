import argparse
import clip
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
import sys

# Import the new data structures
try:
    from data import templates, attributes
except ImportError:
    print("Error: Could not import 'templates' and 'attributes' from data.py.")
    print("Please make sure data.py is in the same directory and contains these variables.")
    sys.exit(1)

# --- 1. CONFIGURATION ---
parser = argparse.ArgumentParser(description="Precompute CLIP embeddings and SAE latents.")
parser.add_argument("--main_task", type=str, default="profession", help="Main task attribute (e.g., 'profession')")
parser.add_argument("--bias_attr", type=str, default="gender", help="Bias attribute (e.g., 'gender' or 'ethnicity')")
parser.add_argument("--clip_model", type=str, default="ViT-B/16", help="CLIP model name to use")

args = parser.parse_args()

MAIN_TASK = args.main_task
BIAS_ATTR = args.bias_attr
CLIP_MODEL_NAME = args.clip_model

# --- Model-specific settings ---
MODEL_CONFIGS = {
    "ViT-L/14@336px": {
        "input_dim": 768,
        "sae_path": "../debiasing_methods/sae_weights/MSAE_ViT-L-14_768_16384_256x512_reverse.pth"
    },
    "ViT-B/16": {
        "input_dim": 512,
        "sae_path": "../debiasing_methods/sae_weights/MSAE_ViT-B-16_512_16384_256x512_reverse.pth"
    }
}

if CLIP_MODEL_NAME not in MODEL_CONFIGS:
    print(f"Error: Config for model '{CLIP_MODEL_NAME}' not found in MODEL_CONFIGS.")
    print("Please add its 'input_dim' and 'sae_path' to the dictionary.")
    sys.exit(1)

config = MODEL_CONFIGS[CLIP_MODEL_NAME]
SAE_INPUT_DIM = config["input_dim"]
SAE_MODEL_PATH = config["sae_path"]

# --- Dynamic Filename Generation ---
SAFE_MODEL_NAME = CLIP_MODEL_NAME.replace("/", "_").replace("@", "_")

# Kept the original filename order (bias then main) so you don't lose access to older files
EMBEDDING_FILE = f"clip_embeddings_{BIAS_ATTR}_{MAIN_TASK}_{SAFE_MODEL_NAME}.npy"
SAE_FILE = f"sae_latents_{BIAS_ATTR}_{MAIN_TASK}_{SAFE_MODEL_NAME}.npy"
LABELS_FILE = f"clip_labels_{BIAS_ATTR}_{MAIN_TASK}_{SAFE_MODEL_NAME}.csv"

print("--- Configuration ---")
print(f"  Main Task:   {MAIN_TASK}")
print(f"  Bias Attr:   {BIAS_ATTR}")
print(f"  CLIP Model:  {CLIP_MODEL_NAME}")
print(f"  SAE Path:    {SAE_MODEL_PATH}")
print(f"  Output (Emb):  {EMBEDDING_FILE}")
print(f"  Output (SAE):  {SAE_FILE}")
print(f"  Output (Labels): {LABELS_FILE}")
print("---------------------")

# --- 2. MODEL DEFINITIONS ---

class MatryoshkaSAE(torch.nn.Module):
    def __init__(self, input_dim=512, latent_dim=16384, topk_levels=[128, 256, 512], weighting='reverse'):
        super().__init__()
        self.bpre = torch.nn.Parameter(torch.zeros(input_dim))
        self.encoder = torch.nn.Linear(input_dim, latent_dim)
        self.decoder = torch.nn.Linear(latent_dim, input_dim, bias=False)
        self.topk_levels = topk_levels
        self.weighting = self._get_weighting(len(topk_levels), weighting)

    def forward(self, x):
        x_centered = x - self.bpre
        h = self.encoder(x_centered)
        losses = []
        for i, k in enumerate(self.topk_levels):
            mask = torch.zeros_like(h)
            topk = torch.topk(h, k=k, dim=1).indices
            mask.scatter_(1, topk, 1.0)
            z = torch.relu(h * mask)
            x_recon = self.decoder(z) + self.bpre
            loss = self.weighting[i] * torch.nn.functional.mse_loss(x_recon, x)
            losses.append(loss)
        return sum(losses)

    def reconstruct(self, x):
        x_centered = x - self.bpre
        h = self.encoder(x_centered)
        z = torch.relu(h)
        return self.decoder(z) + self.bpre

    def _get_weighting(self, h, mode):
        if mode == 'reverse':
            return [float(h - i) for i in range(h)]
        return [1.0] * h
    

def get_clip_text_embedding(text_prompts, model, normalize):
    with torch.no_grad():
        text_tokens = clip.tokenize(text_prompts).to(model.text_projection.device)
        text_embeddings = model.encode_text(text_tokens)
        if normalize:
            text_embeddings /= text_embeddings.norm(dim=-1, keepdim=True)
    return text_embeddings

# --- 3. MODEL LOADING ---

print("\nLoading models...")
device = "cuda" if torch.cuda.is_available() else "cpu"

model, preprocess = clip.load(CLIP_MODEL_NAME, device=device)
model.eval()
print(f"CLIP model '{CLIP_MODEL_NAME}' loaded on device: {device}")

sae_model = MatryoshkaSAE(input_dim=SAE_INPUT_DIM).to(device)
try:
    sae_model.load_state_dict(torch.load(SAE_MODEL_PATH,
                                         map_location=device,
                                         weights_only=True))
    sae_model.eval()
    print(f"Loaded SAE model from: {SAE_MODEL_PATH}")
except FileNotFoundError:
    print(f"Error: SAE model file not found at '{SAE_MODEL_PATH}'")
    sys.exit(1)


# --- 4. PROMPT GENERATION ---

print("\nStarting precomputation...")

all_prompts = []
metadata = []

print("Generating all prompt combinations...")

try:
    bias_list = attributes[BIAS_ATTR]
except KeyError:
    print(f"Error: Attribute '{BIAS_ATTR}' not found in attributes dictionary from data.py")
    sys.exit(1)

try:
    main_list = attributes[MAIN_TASK]
except KeyError:
    print(f"Error: Attribute '{MAIN_TASK}' not found in attributes dictionary from data.py")
    sys.exit(1)

print(f"Using bias attribute: {BIAS_ATTR} ({len(bias_list)} items)")
print(f"Using main task: {MAIN_TASK} ({len(main_list)} items)")

for i_template, template in enumerate(templates):
    for i_bias, val_bias in enumerate(bias_list):
        for i_main, val_main in enumerate(main_list):
            # Assumes data.py still uses {att1} for bias and {att2} for main task
            prompt = template.format(att1=val_bias, att2=val_main)
            all_prompts.append(prompt)
            metadata.append({
                "template": template,
                BIAS_ATTR: val_bias,
                MAIN_TASK: val_main, 
                "template_index": i_template,
                f"{BIAS_ATTR}_index": i_bias, 
                f"{MAIN_TASK}_index": i_main  
            })

print(f"Generated {len(all_prompts)} prompts.")

# --- 5. EMBEDDING & SAE COMPUTATION ---

batch_size = 256 
all_embeddings = []
all_sae_latents = []

print(f"Calculating embeddings and SAE latents in batches of {batch_size}...")

with torch.no_grad():
    for i in tqdm(range(0, len(all_prompts), batch_size)):
        prompt_batch = all_prompts[i : i + batch_size]
        
        embedding_batch = get_clip_text_embedding(prompt_batch, model, normalize=False)
        latent_batch = torch.relu(sae_model.encoder(embedding_batch - sae_model.bpre))
        
        all_embeddings.append(embedding_batch.cpu())
        all_sae_latents.append(latent_batch.cpu())

text_embeddings_tensor = torch.cat(all_embeddings, dim=0)
sae_latents_tensor = torch.cat(all_sae_latents, dim=0)

print(f"All embeddings calculated. Final tensor shape: {text_embeddings_tensor.shape}")
print(f"All SAE latents calculated. Final tensor shape: {sae_latents_tensor.shape}")

# --- 6. SAVING FILES ---

X_embeddings = text_embeddings_tensor.numpy().astype(np.float32)
np.save(EMBEDDING_FILE, X_embeddings)
print(f"Saved embeddings to '{EMBEDDING_FILE}'")

X_sae_latents = sae_latents_tensor.numpy().astype(np.float32)
np.save(SAE_FILE, X_sae_latents)
print(f"Saved SAE latents to '{SAE_FILE}'")

y_labels_df = pd.DataFrame(metadata)
y_labels_df.to_csv(LABELS_FILE, index=False)
print(f"Saved labels to '{LABELS_FILE}'")

print("\n--- Precomputation Complete ---")
