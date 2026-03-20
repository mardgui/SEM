import torch
import os

from .method_utils import CLIPEmbeddingDebiasingMethod, get_clip_text_embedding
from .zsdebias_training import VAEAdaptor, AdaptorDecoder


retrieval_setting_dict = {
    'gender': 'human',  # Demographic bias
    'race': 'human',  # Demographic bias
    'utkrace': 'human'  # Demographic bias
}

zs_classification_setting_dict = {
    'celeba_blond_hair_gender': 'human',  # Demographic bias
    'waterbirds': 'animal'  # Background bias
}


class ZSDebias(CLIPEmbeddingDebiasingMethod):
    def __init__(self, config):
        super().__init__(config)
        self.device = self.model.text_projection.device
        
        # Dimensions
        self.input_dim = self.model.visual.output_dim
        self.hidden_dim = self.input_dim
        self.hidden_num = config.get('hidden_num', 2)
        self.normalize_embeddings = config.get('normalize_embeddings', False)
        
        # Initialize Models
        self.encoder = VAEAdaptor(self.input_dim, self.hidden_dim, self.input_dim, 
                                  depth=self.hidden_num, residual=False).to(self.device)
        self.decoder = AdaptorDecoder(self.hidden_dim, self.hidden_dim, self.input_dim, 
                                      depth=self.hidden_num, residual=False, norm=self.normalize_embeddings).to(self.device)
        
        # Load Weights
        model_name_clean = config['model_ID'].replace('/', '-')
        if config['task'] == 'retrieval':
            bias_type = retrieval_setting_dict[config['bias_attribute']]
        elif config['task'] == 'zs_classification':
            bias_type = zs_classification_setting_dict[config['class_setting']]

        seed = config.get('seed', -1)
        weight_name = f"{model_name_clean}_{bias_type}_zsdebias_{seed}.pth"
        weight_path = os.path.join('debiasing_methods', 'zsdebias_weights', weight_name)
        
        if not os.path.exists(weight_path):
            raise FileNotFoundError(f"ZSDebias weights not found at {weight_path}. Run debiasing_methods/zsdebias_training.py first.")

        print(f"Loading ZSDebias weights from {weight_path}")
        checkpoint = torch.load(weight_path, map_location=self.device)
        self.encoder.load_state_dict(checkpoint['encoder'])
        self.decoder.load_state_dict(checkpoint['decoder'])
        self.encoder.eval()
        self.decoder.eval()

    def debias_text(self, text_prompts, *args, **kwargs):
        # Text is not modified in ZSDebias inference
        return get_clip_text_embedding(text_prompts, self.model, normalize=True).float()

    def debias_images(self, image_embeddings, *args, **kwargs):
        """
        ZSDebias Inference Logic:
        Training objective forces Bias into the 2nd half (z_s).
        Therefore, we must:
        1. Keep 1st Half (z_i - Task Relevant)
        2. Randomize 2nd Half (z_s - Bias)
        """
        image_embeddings = image_embeddings.to(self.device).float()
        
        if self.normalize_embeddings:
            image_embeddings = image_embeddings / image_embeddings.norm(dim=1, keepdim=True)

        with torch.no_grad():
            z, _, _ = self.encoder(image_embeddings)
            
            half_dim = z.shape[1] // 2
            
            # 1. KEEP First Half (Task-Relevant / Neutral)
            z_i = z[:, :half_dim]
            
            # 2. RANDOMIZE Second Half (Bias)
            z_s_noise = torch.randn(z.shape[0], half_dim, device=self.device)
            
            # 3. Concatenate and Decode
            new_z = torch.cat([z_i, z_s_noise], dim=-1)
            recon = self.decoder(new_z)
            
            recon = recon / recon.norm(dim=1, keepdim=True)
            
        return recon
