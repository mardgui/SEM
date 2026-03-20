import clip
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD

from diverse_prompts import *


device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('ViT-B/16', device=device)
model.eval()


def replace_a_with_an(text):
    words = text.split()
    for i in range(len(words) - 1):
        if words[i].lower() == 'a' and words[i + 1][0].lower() in 'aeiou':
            words[i] = 'an'
    return ' '.join(words)


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



def debias_with_sca_b(input_embeddings, neutral_embeddings, bias_embeddings_list, diverse_embeddings, sae_model):
    def _compute_latent(embedding):
        return torch.relu(sae_model.encoder(embedding - sae_model.bpre))
    
    diverse_activations = _compute_latent(diverse_embeddings)
    n_diverse = diverse_activations.shape[0]
    median_diverse_activations = diverse_activations.median(dim=0).values
    latent_size = median_diverse_activations.shape[0]

    flat_bias_embeddings = []
    embeddings_per_bias_class = []

    for bias_embeddings in bias_embeddings_list:
        flat_bias_embeddings.append(bias_embeddings)
        embeddings_per_bias_class.append(len(bias_embeddings))

    latent = _compute_latent(input_embeddings)
    neutral_latent = _compute_latent(neutral_embeddings)
    class_scores = (diverse_activations.unsqueeze(0) < neutral_latent.unsqueeze(1)).sum(dim=1).float() / n_diverse
    bias_activations = _compute_latent(torch.cat(flat_bias_embeddings, dim=0))
    spurious_scores = torch.zeros(latent_size, dtype=torch.float32, device=device)
    
    i = 0
    for n_bc in embeddings_per_bias_class:
        bc_mask = torch.zeros(bias_activations.shape[0], dtype=torch.bool, device=device)
        bc_mask[i:i + n_bc] = 1
        i = i + n_bc

        bc_activations = bias_activations[bc_mask]
        median_bc_activations = bc_activations.median(dim=0, keepdim=True).values
        non_bc_activations = bias_activations[~bc_mask]

        general_bc_scores = ((diverse_activations.unsqueeze(0) < median_bc_activations.unsqueeze(1)).sum(dim=1).float() / n_diverse).squeeze(0)
        specific_bc_scores = ((non_bc_activations.unsqueeze(0) < median_bc_activations.unsqueeze(1)).sum(dim=1).float() / non_bc_activations.shape[0]).squeeze(0)

        bc_scores = torch.minimum(general_bc_scores, specific_bc_scores)
        spurious_scores = torch.maximum(spurious_scores, bc_scores)

    combined_score = (class_scores - spurious_scores + 1) ** 2
    latent = latent * combined_score + (1 - combined_score) * median_diverse_activations.repeat(latent.shape[0], 1)
    debiased = sae_model.decoder(latent) + sae_model.bpre
    return debiased


def debias_with_orth_proj(input_embeddings, bias_embeddings):
    def get_proj_matrix(embeddings):
        tSVD = TruncatedSVD(n_components=len(embeddings))
        tSVD.fit(embeddings)
        basis = tSVD.components_.T.astype(np.float64)

        proj = basis @ basis.T
        proj = np.eye(proj.shape[0], dtype=np.float64) - proj
        return proj

    input_embeddings = F.normalize(input_embeddings, dim=-1).cpu().float().numpy().astype(np.float64)
    bias_embeddings = F.normalize(bias_embeddings, dim=-1).cpu().float().numpy().astype(np.float64)

    P0 = get_proj_matrix(bias_embeddings)
    debiased = torch.from_numpy(input_embeddings @ P0).float().to(device)
    return debiased


template = 'A photo of a {gender}{profession}.'

genders = ['female', 'male']
professions = [
    'Actor', 'Architect', 'Audiologist', 'Author', 'Baker', 'Barber',
    'Blacksmith', 'Bricklayer', 'Bus Driver', 'Butcher', 'Chef', 'Chemist',
    'Cleaner', 'Coach', 'Comedian', 'Computer Programmer', 'Construction Worker',
    'Consultant', 'Counselor', 'Dancer', 'Dentist', 'Designer', 'Dietitian',
    'DJ', 'Doctor', 'Driver', 'Economist', 'Electrician', 'Engineer',
    'Entrepreneur', 'Farmer', 'Florist', 'Graphic Designer', 'Hairdresser',
    'Historian', 'Journalist', 'Judge', 'Lawyer', 'Librarian', 'Magician',
    'Makeup Artist', 'Mathematician', 'Marine Biologist', 'Mechanic', 'Model',
    'Musician', 'Nanny', 'Nurse', 'Optician', 'Painter', 'Pastry Chef',
    'Pediatrician', 'Photographer', 'Plumber', 'Police Officer', 'Politician',
    'Professor', 'Psychologist', 'Real Estate Agent', 'Receptionist', 'Recruiter',
    'Researcher', 'Sailor', 'Salesperson', 'Surveyor', 'Singer', 'Social Worker',
    'Software Developer', 'Statistician', 'Surgeon', 'Teacher', 'Technician',
    'Therapist', 'Tour Guide', 'Translator', 'Vet', 'Videographer', 'Waiter',
    'Writer', 'Zoologist', 'Accountant', 'Astronaut', 'Biologist', 'Carpenter',
    'Civil Engineer', 'Clerk', 'Detective', 'Editor', 'Firefighter', 'Interpreter',
    'Manager', 'Nutritionist', 'Paramedic', 'Pharmacist', 'Physicist', 'Pilot',
    'Reporter', 'Security Guard', 'Scientist', 'Web Developer'
]

all_input_prompts = []
neutral_prompts = []

for gender in genders:
    for profession in professions:
        all_input_prompts.append(template.format(gender=gender + ' ', profession=profession.lower()))

# Add neutral prompts
for profession in professions:
    neutral_prompt = replace_a_with_an(template.format(gender='', profession=profession.lower()))
    all_input_prompts.append(neutral_prompt)
    neutral_prompts.append(neutral_prompt)

neutral_prompts = neutral_prompts * 3

our_bias_prompts = [
    [
        "A portrait of a woman.",
        "A close-up of a woman's face.",
        "A simple headshot of a female.",
        "A woman smiling at the camera.",
        "A woman looking serious.",
        "A person who is female.",
        "A woman with long hair.",
        "A woman wearing makeup.",
        "A woman wearing lipstick.",
        "A young girl.",
        "A teenage girl.",
        "An adult woman.",
        "An elderly woman.",
        "A photo of a mother.",
        "A photo of a daughter.",
        "A woman wearing a dress.",
        "A woman with feminine facial features.",
        "A group of women.",
        "A photo of her.",
        "A woman's face, profile view."
    ],
    [
        "A portrait of a man.",
        "A close-up of a man's face.",
        "A simple headshot of a male.",
        "A man smiling at the camera.",
        "A man looking serious.",
        "A person who is male.",
        "A man with a beard.",
        "A man with a mustache.",
        "A man with short hair.",
        "A young boy.",
        "A teenage boy.",
        "An adult man.",
        "An elderly man.",
        "A photo of a father.",
        "A photo of a son.",
        "A man wearing a suit and tie.",
        "A man with masculine facial features.",
        "A group of men.",
        "A photo of him.",
        "A man's face, profile view."
    ]
]

orth_proj_bias_prompts = [
    "A photo of a female.",
    "A photo of a woman.",
    "A photo of a male.",
    "A photo of a man."
]


with torch.no_grad():
    base_clip_embeddings = get_clip_text_embedding(all_input_prompts, model=model, normalize=False)

    sae_model = MatryoshkaSAE().to(device)
    sae_model.load_state_dict(torch.load('../debiasing_methods/sae_weights/MSAE_ViT-B-16_512_16384_256x512_reverse.pth',
                                         map_location=device,
                                         weights_only=True))
    sae_model.eval()
    our_bias_embeddings = []
    for bias_prompts in our_bias_prompts:
        our_bias_embeddings.append(get_clip_text_embedding(bias_prompts, model=model, normalize=False))
    diverse_embeddings = get_clip_text_embedding(diverse_prompts, model=model, normalize=False)
    neutral_embeddings = get_clip_text_embedding(neutral_prompts, model=model, normalize=False)
    debiased_with_ours = debias_with_sca_b(base_clip_embeddings, neutral_embeddings, our_bias_embeddings, diverse_embeddings, sae_model)

    orth_proj_bias_embeddings = get_clip_text_embedding(orth_proj_bias_prompts, model=model, normalize=False)
    debiased_with_orth_proj = debias_with_orth_proj(base_clip_embeddings, orth_proj_bias_embeddings)

    base_clip_embeddings = F.normalize(base_clip_embeddings, dim=-1).cpu().float().numpy()
    debiased_with_ours = F.normalize(debiased_with_ours, dim=-1).cpu().float().numpy()
    debiased_with_orth_proj = F.normalize(debiased_with_orth_proj, dim=-1).cpu().float().numpy()

pca = PCA(n_components=2, random_state=42)
base_clip_embeddings_pca = pca.fit_transform(base_clip_embeddings)

pca = PCA(n_components=2, random_state=42)
debiased_with_ours_pca = pca.fit_transform(debiased_with_ours)

pca = PCA(n_components=2, random_state=42)
debiased_with_orth_proj_pca = pca.fit_transform(debiased_with_orth_proj)

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# sns.set_theme(style="whitegrid")

df = pd.DataFrame({
    'PC1': np.concat([base_clip_embeddings_pca[:, 0], debiased_with_orth_proj_pca[:, 0], debiased_with_ours_pca[:, 0]]),
    'PC2': np.concat([base_clip_embeddings_pca[:, 1], debiased_with_orth_proj_pca[:, 1], debiased_with_ours_pca[:, 1]]),
    'Gender': (['Female'] * 100 + ['Male'] * 100 + ['Neutral'] * 100) * 3,
    'Profession': professions * 9,
    'Version': ['Original CLIP'] * 300 + ['Orth-Proj'] * 300 + ['SEM$_b$ (ours)'] * 300
})

# g = sns.FacetGrid(df, col='version', hue='gender')
# g.map(sns.scatterplot, 'x', 'y')
# plt.tight_layout()

# output_filename = 'quali.pdf'
# plt.savefig(output_filename, bbox_inches='tight')

# plt.style.use('seaborn-v0_8-paper') # Use a good style
# fig, axes = plt.subplots(1, 3, figsize=(10, 4), sharex=True, sharey=True) # 1 row, 3 columns

# plot_versions = ['Original CLIP', 'Orth-Proj', 'SEM$_b$ (ours)']
# colors = sns.color_palette("Set2", 3) # Two distinct colors for genders
# gender_map = {'F': colors[0], 'M': colors[1], 'Neutral': colors[2]}

# for i, version in enumerate(plot_versions):
#     ax = axes[i]
#     subset_df = df[df['Version'] == version]

#     # Plot each gender separately to control legend entry
#     for gender_val, color_val in gender_map.items():
#         gender_subset = subset_df[subset_df['Gender'] == gender_val]
#         ax.scatter(
#             gender_subset['PC1'],
#             gender_subset['PC2'], 
#             color=color_val, 
#             label=gender_val.capitalize(),
#             alpha=0.7,
#             s=50, # Size of points
#             edgecolors='w', # White edge for better visibility
#             linewidth=0.5
#         )

#     ax.set_title(version, fontsize=18, fontweight='bold', pad=15)
#     ax.set_xlabel('Principal Component 1', fontsize=12)
#     ax.set_ylabel('Principal Component 2', fontsize=12)
#     ax.grid(True, linestyle=':', alpha=0.6)
#     sns.despine(ax=ax, offset=10) # Clean up borders

# # Add a single legend for the entire figure
# handles, labels = axes[0].get_legend_handles_labels() # Get handles/labels from first subplot
# fig.legend(handles, labels, loc='lower left', bbox_to_anchor=(0.0725, 0.14), 
#            title='Gender', frameon=True, fontsize=12, title_fontsize=12, framealpha=0.6)

# # plt.suptitle('PCA of Gendered Profession Embeddings After Debiasing', fontsize=16, fontweight='bold', y=1.02)
# plt.tight_layout(rect=[0, 0, 1, 0.98]) # Adjust layout to make space for suptitle

# output_filename = 'pca_profession_gender.pdf'
# plt.savefig(output_filename, bbox_inches='tight')

# print(f"Plot saved to '{output_filename}'")


# base_clip_embeddings_neutral = base_clip_embeddings[-100:]

# debiased_with_orth_proj_female = debiased_with_orth_proj[:100]
# debiased_with_orth_proj_male = debiased_with_orth_proj[100:-100]

# debiased_with_ours_female = debiased_with_ours[:100]
# debiased_with_ours_male = debiased_with_ours[100:-100]

# --- START OF PLOTTING CHANGES ---

# --- 1. Set global font to serif ---
matplotlib.rcParams.update({
    'font.family': 'serif',
    'mathtext.fontset': 'stix' # Use STIX serif font for math
})

# --- 2. Create the figure (wider and less tall) ---
# --- CHANGED: Shrunk figsize to make fonts relatively larger ---
fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), sharex=True, sharey=True) # Was (18, 4)

plot_versions = ['Original CLIP', 'Orth-Proj', 'SEM$_b$ (ours)']
# --- 3. Use a vibrant, standard color palette ---
colors = sns.color_palette("colorblind", 3) 
gender_map = {'Female': colors[0], 'Male': colors[1], 'Neutral': colors[2]}

for i, version in enumerate(plot_versions):
    ax = axes[i]
    subset_df = df[df['Version'] == version]

    for gender_val, color_val in gender_map.items():
        gender_subset = subset_df[subset_df['Gender'] == gender_val]
        ax.scatter(
            gender_subset['PC1'],
            gender_subset['PC2'], 
            color=color_val, 
            label=gender_val.capitalize(),
            alpha=0.6,
            s=40, # Kept point size
            edgecolors='w',
            linewidth=0.5
        )

    # --- FONT SIZES ARE KEPT LARGE ---
    ax.set_title(version, fontsize=18, fontweight='bold', pad=15)
    ax.set_xlabel('Principal Component 1', fontsize=12)
    ax.set_ylabel('Principal Component 2', fontsize=12)
    ax.grid(True, linestyle=':', alpha=0.6)
    sns.despine(ax=ax, offset=10)
    ax.tick_params(axis='both', which='major', labelsize=12)

# --- 4. Move legend to the right side (font size increased) ---
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, 
           loc='center left',
           bbox_to_anchor=(1, 0.5), 
           title='Gender', 
           frameon=True, 
           fontsize=15,      # Increased from 12
           title_fontsize=16, # Kept large
           framealpha=0.8)

# --- 5. Adjust layout ---
plt.subplots_adjust(
    left=0.05,
    right=0.90,
    bottom=0.20, # Increased bottom margin for label
    top=0.80,    # Increased top margin for title
    wspace=0.1
)

output_filename = 'pca_profession_gender.pdf'
plt.tight_layout()
plt.savefig(output_filename, bbox_inches='tight')

print(f"Plot saved to '{output_filename}'")
