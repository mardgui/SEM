import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, ConcatDataset, random_split, Dataset
from tqdm import trange
import clip

# Optional: WandB for logging
try:
    import wandb
except ImportError:
    wandb = None

# ==========================================
#        DATASET (Local Definition)
# ==========================================

class GenericEmbeddingDataset(Dataset):
    def __init__(self, dataset_path, model_ID, normalize=False):
        """
        Loads embeddings for training ZSDebias. 
        Expects 'CLIP_{model_ID}_embeddings.npy' in the folder.
        """
        self.dataset_path = dataset_path
        self.normalize = normalize
        emb_file = f'CLIP_{model_ID}_embeddings.npy'
        emb_path = os.path.join(dataset_path, emb_file)
        
        if not os.path.exists(emb_path):
            raise FileNotFoundError(f"Embeddings not found at {emb_path}")
            
        print(f"Loading embeddings from {emb_path}...")
        self.embeddings = np.load(emb_path)
        
        # Metadata is not strictly needed for training, but we check for it
        meta_path = os.path.join(dataset_path, 'metadata.feather')
        if os.path.exists(meta_path):
            self.meta_df = pd.read_feather(meta_path)
        else:
            self.meta_df = None

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, index):
        embedding = self.embeddings[index]
        if self.normalize:
            # Safe normalization
            embedding = embedding / (np.linalg.norm(embedding, axis=-1, keepdims=True) + 1e-10)
        
        meta = {} 
        if self.meta_df is not None:
            try:
                meta = self.meta_df.iloc[index]
            except:
                pass
        return meta, embedding

# ==========================================
#        MODEL ARCHITECTURE
# ==========================================

class ResidualBlock(nn.Module):
    def __init__(self, in_dim, out_dim, activation, drop_rate):
        super().__init__()
        self.linear1 = nn.Linear(in_dim, out_dim)
        self.bn1 = nn.BatchNorm1d(out_dim)
        self.linear2 = nn.Linear(out_dim, out_dim)
        self.activation = activation
        self.dropout = nn.Dropout(drop_rate)
        self.need_proj = (in_dim != out_dim)
        if self.need_proj:
            self.proj = nn.Linear(in_dim, out_dim)
    
    def forward(self, x):
        identity = x
        out = self.linear1(x)
        out = self.bn1(out)
        out = self.activation(out)
        out = self.dropout(out)
        out = self.linear2(out)
        if self.need_proj:
            identity = self.proj(identity)
        return out + identity

def make_adaptor_seq(input_dim, hidden_dim, output_dim, depth=3, decoder=False, residual=True, norm=True):
    activation = nn.LeakyReLU(0.3)
    drop_rate = 0.3
    layers = []
    curr_dim = input_dim
    
    # "All adaptors in ZSDebias are implemented as 2-layer MLPs" (Paper 5.1)
    # This loop constructs the hidden layers
    for _ in range(depth - 1):
        if residual:
            layers.append(ResidualBlock(curr_dim, hidden_dim, activation, drop_rate))
        else:
            # "Each layer is followed by LayerNorm and Leaky ReLU" (Paper 5.1)
            layers.extend([
                nn.Linear(curr_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.LeakyReLU(0.3),
                nn.Dropout(drop_rate),
            ])
        curr_dim = hidden_dim
        
    layers.append(nn.Linear(curr_dim, output_dim))
    if decoder and norm:
        layers.append(nn.Tanh())
    return nn.Sequential(*layers)

class VAEAdaptor(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, depth=3, residual=True):
        super().__init__()
        # Encoder outputs 2 * latent_dim (mu and logvar)
        self.encoder = make_adaptor_seq(input_dim, hidden_dim, 2 * latent_dim, depth, residual=residual)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm1d, nn.LayerNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        h = self.encoder(x)
        mu, logvar = torch.chunk(h, 2, dim=-1)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

class AdaptorDecoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, depth=3, residual=True, norm=True):
        super().__init__()
        self.decoder = make_adaptor_seq(input_dim, hidden_dim, output_dim, depth, decoder=True, residual=residual, norm=norm)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm1d, nn.LayerNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.decoder(x)

# ==========================================
#              LOSS & UTILS
# ==========================================

def centering(K):
    n = K.shape[0]
    unit = torch.ones(n, n, device=K.device) / n
    I = torch.eye(n, device=K.device)
    H = I - unit
    return H @ K @ H

def rbf(X, sigma=None):
    GX = X @ X.T
    sq_dists = GX.diag().unsqueeze(0) - 2 * GX + GX.diag().unsqueeze(1)
    if sigma is None:
        mdist = sq_dists[sq_dists != 0].median()
        sigma = torch.sqrt(mdist)
    KX = torch.exp(-sq_dists / (2 * sigma * sigma))
    return KX

def kernel_CKA(X, Y, sigma=None):
    eps = 1e-8
    K = rbf(X, sigma)
    L = rbf(Y, sigma)
    K_center = centering(K)
    L_center = centering(L)
    hsic_xy = torch.sum(K_center * L_center)
    norm_X = torch.sqrt(torch.sum(K_center * K_center) + eps)
    norm_Y = torch.sqrt(torch.sum(L_center * L_center) + eps)
    return hsic_xy / (norm_X * norm_Y)

def set_seeds(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class EarlyStopper:
    """Early stopping based on validation loss saturation (Paper 5.1)"""
    def __init__(self, patience=20, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
            return False, True # Stop=False, Save=True (New Best)
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True, False # Stop=True
        return False, False # Continue

# ==========================================
#        BIAS CORPUS GENERATION
# ==========================================

def get_prefixes():
    """Table 6 from Supplementary Material"""
    return [
        "snapshot of", "portrait of", "image of",
        "depiction of", "rendering of", "illustration of",
        "capture of", "representation of", "scene of",
        "view of", "glimpse of", "close-up of",
        "aerial view of", "detailed look at", "panoramic view of",
        "sketch of", "digital creation of", "vivid depiction of",
        "artistic interpretation of", "snapshot capturing",
        # Adding standard CLIP templates for robustness
        "a photo of", "a picture of", "an image of", "a low resolution photo of",
        "a good photo of", "a bad photo of", "a cropped photo of"
    ]

def get_demographic_terms():
    """Tables 7, 8, 9, 11 from Supplementary Material"""
    gender = [
        "Male", "Female", "Man", "Woman", "Boy", "Girl",
        "Guy", "Gal", "Gentleman", "Lady", "Dude", "Chick",
        "Father", "Mother", "Son", "Daughter", "Brother", "Sister",
        "Husband", "Wife", "Uncle", "Aunt", "Nephew", "Niece",
        "Grandfather", "Grandmother", "Groom", "Bride", "Father-in-law", "Mother-in-law",
        "Son-in-law", "Daughter-in-law", "Stepfather", "Stepmother", "Stepson", "Stepdaughter",
        "Godfather", "Godmother", "Godson", "Goddaughter", "Boyfriend", "Girlfriend",
        "Fiancé", "Fiancée", "Bachelor", "Bachelorette", "Widower", "Widow",
        "King", "Queen", "Prince", "Princess", "Actor", "Actress",
        "Waiter", "Waitress", "Host", "Hostess", "Master", "Mistress",
        "Lord", "Lady", "Duke", "Duchess", "Emperor", "Empress",
        "Count", "Countess", "Sir", "Madam"
    ]
    
    age = [
        "Newborn", "Infant", "Toddler", "Preschooler", "Preteen", "Adolescent",
        "Teenager", "Octogenarian", "Nonagenarian", "Centenarian", "Quadragenarian", "Quinquagenarian",
        "Sexagenarian", "Septuagenarian", "Tween", "Youngster", "Juvenile", "Minor",
        "Adult", "Senior", "Geriatric", "Neonate", "Weanling", "Yearling",
        "Fledgling", "Pubescent", "Postpubescent", "Middlescent", "Supercentenarian", "Dotard",
        "Venerable Youth", "Elder", "Golden-ager", "Retiree", "Pensioner",
        "Middle-ager", "Thirtysomething", "Fortysomething", "Fiftysomething", "Sixtysomething",
        "Silver", "Codger", "Cradle", "Fogy", "Ancestor", "Descendant", "Baby", "Child", "Kid",
        "Teen", "Young adult", "Middle-aged", "Elderly", "Ancient", "Millennial",
        "Generation Z", "Baby boomer", "Generation X"
    ]
    
    race = [
        "Caucasian", "African", "Asian", "Hispanic", "Indigenous", "Aboriginal",
        "Maori", "Sami", "Inuit", "Berber", "Polynesian", "Melanesian",
        "Micronesian", "Amerindian", "Mestizo", "Mulatto", "Creole", "Romani",
        "Bedouin", "Tuareg", "Fulani", "Yoruba", "Zulu", "Xhosa",
        "Masai", "Han", "Bengali", "Tamil", "Sikh", "Malay",
        "Ainu", "Tibetan", "Uyghur", "Kurd", "Pashtun", "Hazara",
        "Tajik", "Baloch", "Sinhalese", "Karen", "Hmong", "Maasai",
        "San", "Pygmy", "Bantu", "Aztec", "Maya", "Inca",
        "Mapuche", "Quechua", "Aymara", "Métis", "First Nations", "Aborigine"
    ]
    
    appearance = [
        "Person with albinism", "Person with freckles", "Person with vitiligo", "Person with alopecia",
        "Hairy person", "Person with epicanthic folds", "Person with brachia", "Right-handed person",
        "Left-handed person", "Person with amputation", "Person with visual impairment", "Person with hearing impairment",
        "Person with muteness", "Able-bodied person", "Neurotypical person", "Person with amblyopia",
        "Tattooed person", "Pierced person", "Scarred person", "Burn victim",
        "Person using mobility aids", "Person using prosthetics", "Person using wheelchair", "Person with orthopedic condition",
        "Person with dwarfism", "Person with gigantism", "Pigmented person", "Person with melanism",
        "Androgynous person", "Intersex person", "Person with hemophilia", "Person with diabetes",
        "Person with asthma", "Person with allergies", "Person with autism", "Person with dyslexia",
        "Ambidextrous person", "Eunuch", "Person with kyphosis", "Person with lactose intolerance", 
        "Person with celiac disease", "Person with epilepsy", "Person with paraplegia", "Person with quadriplegia", 
        "Person with hemiplegia", "Person with strabismus", "Person with tremors", "Person with nystagmus", 
        "Person with dysphonia", "Person who stutters"
    ]

    return gender + age + race + appearance

def get_background_descriptions():
    """Table 15 from Supplementary Material"""
    return [
        "A forest with tall trees", "A beach with clear waters", "A snow-covered mountain", "A lakeside with dense foliage",
        "A coral reef underwater", "A thunderstorm with lightning", "A pond with lily pads", "A view of the Grand Canyon",
        "A foggy morning in a forest", "A starry night in the wilderness", "A forest glade with deer", "A snowy landscape with a cabin",
        "A colorful autumn forest", "A secluded beach with a cove", "A tropical paradise with palms", "A rugged canyon with cliffs",
        "A serene waterfall in a canyon", "A river winding through a valley", "A green pasture with grazing animals", "A vast savanna with wildlife",
        "A field ready for harvest", "A city park with greenery", "A bamboo forest with winding paths", "An ocean under a clear sky",
        "A desert at sunset", "A dense jungle with a waterfall", "A tundra covered in snow and ice", "A clear stream in a quiet glen",
        "A rocky coastline with waves", "A quiet pond surrounded by trees", "A redwood forest with tall trees", "A sunflower field in sunlight",
        "An alpine lake reflecting mountains", "A coral reef with marine life", "A meadow with birds", "A vineyard with grapevines",
        "A pagoda surrounded by blossoms", "A waterfall in a rainforest", "A lavender field in full bloom", "A glacial lake with clear waters",
        "A city square with historic architecture", "A sun-soaked beach with palm trees", "A pine forest with needles", "A bustling harbor with boats",
        "A meadow with wildflowers", "A pond with water lilies", "A bayou with cypress trees", "A waterfall with a rainbow",
        "An olive grove under the sun", "A hillside with vineyards", "A mangrove forest by the coastline", "A koi pond in a garden",
        "A cottage in a vineyard", "A lavender field with purple blooms", "A forest glen with a brook", "A beach with footprints in the sand",
        "An urban park with families", "A garden with a stone fountain", "A fern-covered forest floor", "A lake at dawn with mist",
        "A marsh with tall reeds", "A garden with butterflies", "A garden with a wooden bridge", "A mangrove swamp with waterways",
        "A cottage in a garden", "A lavender field with fragrant flowers", "A beach with seashells", "A market square with vendors",
        "A garden with a pond", "A fern-covered forest floor", "A lake at sunset with reflections", "A marshland with tall grasses",
        "A field of poppies in full bloom", "A desert landscape under a starry sky", "A village nestled in a valley", "A cityscape with skyscrapers",
        "A forest path with fallen leaves", "A mountain peak at sunrise", "A river cutting through a dense forest", "A tranquil beach at dusk",
        "A misty valley at dawn", "A desert oasis with palm trees", "A volcanic landscape with steam vents", "A rice terrace in morning light",
        "A cherry blossom garden in spring", "A medieval castle on a hilltop", "A lighthouse on a rocky shore", "An ancient temple in ruins",
        "A cobblestone street in old town", "A mountain stream with rapids", "A field of tulips in bloom", "A zen garden with raked sand",
        "A mountain peak at sunrise", "A river cutting through a dense forest", "A tranquil beach at dusk", "A snowy street in a small town",
        "An ancient temple in a jungle", "A desert oasis with palm trees", "A flower garden in full bloom", "A frozen lake surrounded by pine trees",
        "A cave entrance surrounded by vines", "A country road lined with autumn trees", "A traditional village with thatched roofs", "A deep forest trail with sunlight filtering through",
        "A snowy village with lights", "A castle ruin on a hill", "A serene pond with ducks"
    ]

def generate_full_bias_corpus(args, device):
    print(f"Generating bias corpus for attribute: {args.bias_attribute}...")
    
    prefixes = get_prefixes()
    prompts = []

    if args.bias_attribute in ['human', 'gender', 'race']: 
        # Human Bias: Prefix + "a" + Term
        terms = get_demographic_terms()
        print(f"Combining {len(prefixes)} prefixes with {len(terms)} demographic terms...")
        
        for prefix in prefixes:
            for term in terms:
                # Example: "portrait of a Male"
                # We handle the article "a/an" loosely or add it explicitly
                prompts.append(f"{prefix} a {term}")
                prompts.append(f"{prefix} {term}")

    elif args.bias_attribute == 'animal':
        # Background/Animal Bias: Prefix + Description
        # Note: Background descriptions in Table 15 already start with "A ..." or "An ..."
        # e.g., "A forest with tall trees"
        backgrounds = get_background_descriptions()
        print(f"Combining {len(prefixes)} prefixes with {len(backgrounds)} background descriptions...")
        
        for prefix in prefixes:
            for bg in backgrounds:
                # Example: "snapshot of" + "A forest..." -> "snapshot of A forest..."
                prompts.append(f"{prefix} {bg}")
                # Lowercase version: "snapshot of a forest..."
                prompts.append(f"{prefix} {bg.lower()}")

    else:
        print(f"Warning: No preset prompts for {args.bias_attribute}. Using generic template.")
        base_prompts = [f"a photo of a {args.bias_attribute}", f"a {args.bias_attribute} person"]
        prompts = base_prompts # Fallback

    # Deduplicate
    prompts = list(set(prompts))
    print(f"Total corpus size: {len(prompts)} prompts.")
    
    # Compute Embeddings
    model, _ = clip.load(args.clip_model, device=device)
    all_embeddings = []
    batch_size = 512 # Increased batch size for speed
    
    with torch.no_grad():
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i+batch_size]
            tok = clip.tokenize(batch, truncate=True).to(device)
            emb = model.encode_text(tok)
            emb = emb / emb.norm(dim=1, keepdim=True)
            all_embeddings.append(emb)
            
    return torch.cat(all_embeddings).float()

# ==========================================
#             TRAINING LOOP
# ==========================================

def train(args):
    device = torch.device(args.device)
    set_seeds(args.seed)

    if args.use_wandb and wandb is not None:
        wandb.init(
            project=f"{args.clip_model.replace('/', '-')}_{args.bias_attribute}_zsdebias",
            config=vars(args),
            name=f"{args.bias_attribute}_kl{args.kl_weight}_cka{args.cka_weight}"
        )

    print("Loading datasets...")
    datasets_list = []
    model_id_clean = args.clip_model.replace('/', '-')
    
    for path in args.dataset_paths:
        if os.path.exists(path):
            ds = GenericEmbeddingDataset(dataset_path=path, model_ID=model_id_clean, normalize=False)
            datasets_list.append(ds)
        else:
            print(f"Warning: Path {path} does not exist.")
            
    if not datasets_list:
        raise ValueError("No valid datasets loaded. Check --dataset_paths.")
        
    full_dataset = ConcatDataset(datasets_list)
    
    # 70% Training / 30% Validation as per paper (Section 5.1)
    n_total = len(full_dataset)
    n_train = int(0.7 * n_total) 
    n_val = n_total - n_train
    
    train_subset, val_subset = random_split(full_dataset, [n_train, n_val])
    print(f"Images: Total {n_total} | Train {n_train} | Val {n_val}")

    def embedding_collate(batch):
        embeddings = [torch.tensor(item[1]) for item in batch]
        return torch.stack(embeddings).float()

    train_loader = DataLoader(
        train_subset, batch_size=args.bs, shuffle=True, 
        num_workers=args.num_workers, collate_fn=embedding_collate, drop_last=True
    )
    val_loader = DataLoader(
        val_subset, batch_size=args.bs, shuffle=False, 
        num_workers=args.num_workers, collate_fn=embedding_collate, drop_last=False
    )
    
    bias_embedd = generate_full_bias_corpus(args, device)
    
    # Determine input dimension from data
    input_dim = datasets_list[0].embeddings.shape[1]
    
    # FIX: Overwrite hidden_dim to match CLIP embeddings (Paper 5.1)
    hidden_dim = input_dim
    
    encoder = VAEAdaptor(input_dim, hidden_dim, input_dim, depth=args.hidden_num, residual=False).to(device)
    decoder = AdaptorDecoder(hidden_dim, hidden_dim, input_dim, depth=args.hidden_num, residual=False, norm=args.normalize_embeddings).to(device)
    
    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()),
        lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-8)
    
    # Paper 5.1: "implemented early stopping base on the saturation of the validation loss"
    early_stopper = EarlyStopper(patience=args.patience)
    
    print("Starting training...")
    cycle_length = 50 
    metrics_keys = ['total_loss', 'recon_loss', 'cka_loss', 'irrel_loss', 'cont_loss', 'kl_loss']
    
    save_dir = 'debiasing_methods/zsdebias_weights'
    os.makedirs(save_dir, exist_ok=True)
    save_name = f"{model_id_clean}_{args.bias_attribute}_zsdebias_{args.seed}.pth"
    save_path = os.path.join(save_dir, save_name)

    for epoch in trange(1, args.epochs + 1):
        # --- TRAIN ---
        encoder.train()
        decoder.train()
        train_metrics = {k: 0.0 for k in metrics_keys}
        steps = 0
        annealed_kl_weight = 1e-2 * ((epoch - 1) % cycle_length) / (cycle_length - 1)
        
        for image_embeddings in train_loader:
            image_embeddings = image_embeddings.to(device)
            if args.normalize_embeddings:
                image_embeddings = image_embeddings / image_embeddings.norm(dim=1, keepdim=True)
                
            optimizer.zero_grad()
            z, mu, logvar = encoder(image_embeddings)
            
            half_dim = z.size(1) // 2
            z_i = z[:, :half_dim]
            z_s = z[:, half_dim:] 
            
            recon_embed = decoder(z)
            if args.normalize_embeddings:
                recon_embed = recon_embed / recon_embed.norm(dim=1, keepdim=True)
                
            sim_base = image_embeddings if args.normalize_embeddings else image_embeddings / image_embeddings.norm(dim=1, keepdim=True)
            sim_scores = sim_base @ bias_embedd.t()
            _, top_k_indices = sim_scores.topk(args.top_k, dim=1)
            ensemble_embeds = bias_embedd[top_k_indices].mean(dim=1)
            ensemble_embeds = ensemble_embeds / ensemble_embeds.norm(dim=1, keepdim=True)
            
            cka_loss = kernel_CKA(z_s, ensemble_embeds)
            irrel_loss = kernel_CKA(z_i, z_s)
            recon_loss = F.mse_loss(recon_embed, image_embeddings, reduction='sum') / image_embeddings.size(0)
            kl_loss = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
            
            idx = torch.randperm(image_embeddings.size(0))
            z_s_shuffled = z_s[idx]
            z_i_shuffled = z_i[idx] 
            
            recon_noise_s = decoder(torch.cat([z_s_shuffled, z_i], dim=-1))
            recon_noise_i = decoder(torch.cat([z_s, z_i_shuffled], dim=-1))
            
            if args.normalize_embeddings:
                recon_noise_s = recon_noise_s / recon_noise_s.norm(dim=1, keepdim=True)
                recon_noise_i = recon_noise_i / recon_noise_i.norm(dim=1, keepdim=True)
            
            cont_loss = (F.mse_loss(recon_noise_s, image_embeddings, reduction='sum') + 
                         F.mse_loss(recon_noise_i, image_embeddings, reduction='sum')) / image_embeddings.size(0)

            loss = (args.recon_weight * recon_loss - 
                    args.cka_weight * cka_loss + 
                    args.irrel_weight * irrel_loss + 
                    args.cont_weight * cont_loss + 
                    annealed_kl_weight * kl_loss)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(encoder.parameters()) + list(decoder.parameters()), 1.0)
            optimizer.step()
            
            train_metrics['total_loss'] += loss.item()
            train_metrics['recon_loss'] += recon_loss.item()
            train_metrics['cka_loss'] += cka_loss.item()
            train_metrics['irrel_loss'] += irrel_loss.item()
            train_metrics['cont_loss'] += cont_loss.item()
            train_metrics['kl_loss'] += kl_loss.item()
            steps += 1
            
        scheduler.step()
        
        # --- VALIDATION ---
        encoder.eval()
        decoder.eval()
        val_metrics = {k: 0.0 for k in metrics_keys}
        val_steps = 0
        
        with torch.no_grad():
            for image_embeddings in val_loader:
                image_embeddings = image_embeddings.to(device)
                if args.normalize_embeddings:
                    image_embeddings = image_embeddings / image_embeddings.norm(dim=1, keepdim=True)
                
                z, mu, logvar = encoder(image_embeddings)
                half_dim = z.size(1) // 2
                z_i = z[:, :half_dim]
                z_s = z[:, half_dim:]
                
                recon_embed = decoder(z)
                if args.normalize_embeddings:
                    recon_embed = recon_embed / recon_embed.norm(dim=1, keepdim=True)
                    
                sim_base = image_embeddings if args.normalize_embeddings else image_embeddings / image_embeddings.norm(dim=1, keepdim=True)
                sim_scores = sim_base @ bias_embedd.t()
                _, top_k_indices = sim_scores.topk(args.top_k, dim=1)
                ensemble_embeds = bias_embedd[top_k_indices].mean(dim=1)
                ensemble_embeds = ensemble_embeds / ensemble_embeds.norm(dim=1, keepdim=True)
                
                cka_loss = kernel_CKA(z_s, ensemble_embeds)
                irrel_loss = kernel_CKA(z_i, z_s)
                recon_loss = F.mse_loss(recon_embed, image_embeddings, reduction='sum') / image_embeddings.size(0)
                kl_loss = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
                
                idx = torch.randperm(image_embeddings.size(0))
                z_s_shuffled = z_s[idx]
                z_i_shuffled = z_i[idx] 
                
                recon_noise_s = decoder(torch.cat([z_s_shuffled, z_i], dim=-1))
                recon_noise_i = decoder(torch.cat([z_s, z_i_shuffled], dim=-1))
                
                if args.normalize_embeddings:
                    recon_noise_s = recon_noise_s / recon_noise_s.norm(dim=1, keepdim=True)
                    recon_noise_i = recon_noise_i / recon_noise_i.norm(dim=1, keepdim=True)
                
                cont_loss = (F.mse_loss(recon_noise_s, image_embeddings, reduction='sum') + 
                             F.mse_loss(recon_noise_i, image_embeddings, reduction='sum')) / image_embeddings.size(0)
                
                loss = (args.recon_weight * recon_loss - 
                        args.cka_weight * cka_loss + 
                        args.irrel_weight * irrel_loss + 
                        args.cont_weight * cont_loss + 
                        annealed_kl_weight * kl_loss)
                
                val_metrics['total_loss'] += loss.item()
                val_metrics['recon_loss'] += recon_loss.item()
                val_metrics['cka_loss'] += cka_loss.item()
                val_metrics['irrel_loss'] += irrel_loss.item()
                val_metrics['cont_loss'] += cont_loss.item()
                val_metrics['kl_loss'] += kl_loss.item()
                val_steps += 1
        
        # Log & Print
        log_dict = {}
        for k in metrics_keys:
            log_dict[f"[Train] {k}"] = train_metrics[k] / steps
            log_dict[f"[Valid] {k}"] = val_metrics[k] / val_steps
        
        if args.use_wandb and wandb is not None:
            wandb.log(log_dict)
            
        # if epoch % 10 == 0:
        print(f"Epoch {epoch}: Val Loss {log_dict['[Valid] total_loss']:.4f} (CKA: {log_dict['[Valid] cka_loss']:.4f})")

        # Early Stopping & Saving
        stop_training, save_checkpoint = early_stopper.early_stop(log_dict['[Valid] total_loss'])
        
        if save_checkpoint:
            torch.save({
                'encoder': encoder.state_dict(),
                'decoder': decoder.state_dict(),
                'args': vars(args),
                'epoch': epoch,
                'val_loss': log_dict['[Valid] total_loss']
            }, save_path)
            # if epoch % 10 == 0:
            print(f"--> Best model saved to {save_path}")
        
        if stop_training:
            print(f"Early stopping triggered at epoch {epoch}")
            break

    print(f"Training complete. Best model available at {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_paths", nargs='+', required=True, help="Paths to Generic Vision Datasets")
    parser.add_argument("--bias_attribute", type=str, default="human", help="Bias type (human, animal)")
    parser.add_argument("--clip_model", type=str, default="ViT-L/14")
    parser.add_argument("--hidden_num", type=int, default=2)
    parser.add_argument("--normalize_embeddings", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--bs", type=int, default=1024)
    
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    
    parser.add_argument("--recon_weight", type=float, default=100.0)
    parser.add_argument("--cka_weight", type=float, default=10.0)
    parser.add_argument("--irrel_weight", type=float, default=10.0)
    parser.add_argument("--cont_weight", type=float, default=10.0)
    
    parser.add_argument("--kl_weight", type=float, default=1e-3)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--double_cont", action="store_true", default=True)
    parser.add_argument("--augmentation_factor", type=int, default=5)
    
    parser.add_argument("--patience", type=int, default=50, help="Early stopping patience")
    
    parser.add_argument("--use_wandb", action="store_true", help="Enable WandB logging")

    args = parser.parse_args()
    train(args)
