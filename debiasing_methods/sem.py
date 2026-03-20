import os

from .method_utils import *
from .sem_data.diverse_prompts import diverse_prompts


class MatryoshkaSAE(torch.nn.Module):
    def __init__(self, input_dim, latent_dim, topk_levels, weighting):
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


class SEM(CLIPEmbeddingDebiasingMethod):
    def __init__(self, config: dict):
        super().__init__(config)
        self.version = config['sem_version']
        self.device = self.model.text_projection.device

        model_path = config['sae_weights_path']
        _, _, input_dim, latent_dim, granularities, weighting = os.path.basename(model_path).split('_')
        input_dim = int(input_dim)
        latent_dim = int(latent_dim)
        self.latent_dim = latent_dim
        topk_levels = [int(g) for g in granularities.split('x')]
        self.sae_model = MatryoshkaSAE(input_dim, latent_dim, topk_levels, weighting).to(self.device)
        self.sae_model.load_state_dict(
            torch.load(model_path, map_location=self.device, weights_only=True)
        )
        self.sae_model.eval()

        with torch.no_grad():
            diverse_embeddings = get_clip_text_embedding(diverse_prompts, self.model, normalize=False).float()
            self.diverse_activations = self._compute_latent(diverse_embeddings)
            self.n_diverse = self.diverse_activations.shape[0]
            self.median_diverse_activations = self.diverse_activations.median(dim=0).values
            
    def _compute_latent(self, embedding):
        latent = torch.relu(self.sae_model.encoder(embedding - self.sae_model.bpre))
        return latent

    def debias_text(self, text_prompts, *args, **kwargs):
        extra_prompts = kwargs['extra_prompts']

        if self.version in ['SEM_b', 'SEM_bi']:
            bias_prompts = []
            prompts_per_bias_class = []
            for bias_class in extra_prompts['bias prompts']:
                bc_prompts = extra_prompts['bias prompts'][bias_class]
                bias_prompts.extend(bc_prompts)
                prompts_per_bias_class.append(len(bc_prompts))

        if self.version in ['SEM_i', 'SEM_bi']:
            extra_class_prompts = []
            prompts_per_class = []

            for class_prompts in extra_prompts['extra class prompts']:
                extra_class_prompts.extend(class_prompts)
                prompts_per_class.append(len(class_prompts))
        
        with torch.no_grad():
            if self.version in ['SEM_b', 'SEM_bi']:
                query_embedding = get_clip_text_embedding(text_prompts, self.model, normalize=False).float()
                bias_embeddings = get_clip_text_embedding(bias_prompts, self.model, normalize=False).float()
            if self.version in ['SEM_i', 'SEM_bi']:
                extra_class_embeddings = get_clip_text_embedding(extra_class_prompts, self.model, normalize=False).float()

                extra_class_activations = self._compute_latent(extra_class_embeddings)
                median_class_activations = []
            
                i = 0
                for n_tc in prompts_per_class:
                    target_class_activations = extra_class_activations[i:i + n_tc]
                    median_class_activations.append(target_class_activations.median(dim=0, keepdim=True).values)
                    i = i + n_tc

            if self.version in ['SEM_b', 'SEM_bi']:
                latent = self._compute_latent(query_embedding)

            if self.version == 'SEM_i':
                latent = torch.cat(median_class_activations, dim=0)

            if self.version == 'SEM_bi':
                median_class_activations = torch.cat(median_class_activations, dim=0)
                class_scores = (self.diverse_activations.unsqueeze(0) < median_class_activations.unsqueeze(1)).sum(dim=1).float() / self.n_diverse
            
            if self.version in ['SEM_b', 'SEM_i']:
                class_scores = (self.diverse_activations.unsqueeze(0) < latent.unsqueeze(1)).sum(dim=1).float() / self.n_diverse

            if self.version in ['SEM_b', 'SEM_bi']:
                bias_activations = self._compute_latent(bias_embeddings)
                bias_scores = torch.zeros(self.latent_dim, dtype=torch.float32, device=self.device)

                i = 0
                for n_bc in prompts_per_bias_class:
                    bc_mask = torch.zeros(bias_activations.shape[0], dtype=torch.bool, device=self.device)
                    bc_mask[i:i + n_bc] = 1
                    i = i + n_bc

                    bc_activations = bias_activations[bc_mask]
                    median_bc_activations = bc_activations.median(dim=0, keepdim=True).values
                    non_bc_activations = bias_activations[~bc_mask]

                    general_bc_scores = ((self.diverse_activations.unsqueeze(0) < median_bc_activations.unsqueeze(1)).sum(dim=1).float() / self.n_diverse).squeeze(0)
                    specific_bc_scores = ((non_bc_activations.unsqueeze(0) < median_bc_activations.unsqueeze(1)).sum(dim=1).float() / non_bc_activations.shape[0]).squeeze(0)

                    bc_scores = torch.minimum(general_bc_scores, specific_bc_scores)
                    bias_scores = torch.maximum(bias_scores, bc_scores)
                
            if self.version == 'SEM_i':
                combined_score = class_scores ** 2

            if self.version in ['SEM_b', 'SEM_bi']:
                combined_score = (1 + class_scores - bias_scores) ** 2

            latent = latent * combined_score + (1 - combined_score) * self.median_diverse_activations.repeat(latent.shape[0], 1)
            debiased = self.sae_model.decoder(latent) + self.sae_model.bpre

        return debiased

    def debias_images(self, image_embeddings, *args, **kwargs):
        return image_embeddings.float()
