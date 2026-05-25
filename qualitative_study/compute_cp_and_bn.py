import clip
import numpy as np
import torch
import torch.nn.functional as F
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

# =====================================================================
# Metric Evaluation Script (Content Preservation & Bias Neutralization)
# =====================================================================

def evaluate_metrics(debiased_embeddings, base_clip_embeddings, n_professions=100):
    """
    Computes Content Preservation (CP) and Bias Neutralization (BN) metrics.
    
    Expected indexing of embeddings array:
      - 0 to n_professions-1: Female debiased
      - n_professions to 2*n_professions-1: Male debiased
      - 2*n_professions to 3*n_professions-1: Neutral debiased
    """
    # 1. Extract gendered debiased embeddings
    z_female = debiased_embeddings[0:n_professions]
    z_male = debiased_embeddings[n_professions:2*n_professions]
    
    # 2. Extract the ORIGINAL Base CLIP neutral anchor embeddings
    # (Indices 200 to 299 correspond to the unique neutral prompts)
    z_neutral_orig = base_clip_embeddings[2*n_professions:3*n_professions]
    
    # ---- Bias Neutralization (BN) ----
    # BN = (1 / |P|) * sum( cos(z_male, z_female) )
    # Since arrays are L2-normalized, the row-wise dot product is the cosine similarity
    bn_per_profession = np.sum(z_male * z_female, axis=1)
    bn = np.mean(bn_per_profession)
    
    # ---- Content Preservation (CP) ----
    # CP = (1 / (|P| * |G|)) * sum_p sum_g cos(z_g, z_neutral_orig)
    cp_female = np.sum(z_female * z_neutral_orig, axis=1)
    cp_male = np.sum(z_male * z_neutral_orig, axis=1)
    cp = np.mean(np.concatenate([cp_female, cp_male]))
    
    return cp, bn


# Compute metrics for all configurations
n_profs = len(professions)

cp_base, bn_base = evaluate_metrics(base_clip_embeddings, base_clip_embeddings, n_professions=n_profs)
cp_orth, bn_orth = evaluate_metrics(debiased_with_orth_proj, base_clip_embeddings, n_professions=n_profs)
cp_ours, bn_ours = evaluate_metrics(debiased_with_ours, base_clip_embeddings, n_professions=n_profs)

# Print a structured markdown table matching your paper's format
print("\n" + "="*50)
print("             EVALUATION RESULTS")
print("="*50)
print(f"{'Method':<15} | {'Content Preservation (↑)':<24} | {'Bias Neutralization (↑)':<23}")
print("-" * 70)
print(f"{'Base CLIP':<15} | {cp_base:<24.3f} | {bn_base:<23.3f}")
print(f"{'ORTH-PROJ':<15} | {cp_orth:<24.3f} | {bn_orth:<23.3f}")
print(f"{'SEMb (Ours)':<15} | {cp_ours:<24.3f} | {bn_ours:<23.3f}")
print("="*50)
