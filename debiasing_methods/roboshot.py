import torch.nn.functional as F

from .method_utils import *


class RoboShot(CLIPEmbeddingDebiasingMethod):
    def __init__(self, config):
        super().__init__(config)
        self.reject_harmful = config['reject_harmful']
        self.accept_helpful = config['accept_helpful']

    def debias_text(self, text_prompts, *args, **kwargs):
        with torch.no_grad():
            query_embedding = get_clip_text_embedding(text_prompts, self.model, normalize=False).float()
        return query_embedding

    def debias_images(self, image_embeddings, *args, **kwargs):
        extra_prompts = kwargs['extra_prompts']
        device = self.model.text_projection.device

        harmful_prompts = np.array(extra_prompts['harmful prompts'])
        harmful_prompts_shape = (*harmful_prompts.shape, -1)
        flat_harmful_prompts = harmful_prompts.flatten()
        helpful_prompts = np.array(extra_prompts['helpful prompts'])
        helpful_prompts_shape = (*helpful_prompts.shape, -1)
        flat_helpful_prompts = helpful_prompts.flatten()
        
        with torch.no_grad():
            flat_harmful_embeddings = get_clip_text_embedding(flat_harmful_prompts, self.model, normalize=False).cpu().float().numpy().astype(np.float64)
            flat_helpful_embeddings = get_clip_text_embedding(flat_helpful_prompts, self.model, normalize=False).cpu().float().numpy().astype(np.float64)
            image_embeddings = F.normalize(image_embeddings, dim=-1).cpu().float().numpy().astype(np.float64)

        harmful_embeddings = flat_harmful_embeddings.reshape(harmful_prompts_shape)
        helpful_embeddings = flat_helpful_embeddings.reshape(helpful_prompts_shape)

        if self.reject_harmful:
            spurious_vectors = harmful_embeddings[:, 0] - harmful_embeddings[:, 1]
            q_spurious, _ = np.linalg.qr(spurious_vectors.T)
            q_spurious = q_spurious.T

            for orthonormal_vector in q_spurious:
                sim = image_embeddings @ orthonormal_vector
                harmful_features = sim[:, None] * np.repeat(orthonormal_vector[None, :], sim.shape[0], axis=0) / np.linalg.norm(orthonormal_vector)
                image_embeddings = image_embeddings - harmful_features
                image_embeddings = image_embeddings / np.linalg.norm(image_embeddings, axis=1, keepdims=True)

        if self.accept_helpful:
            true_vectors = helpful_embeddings[:, 0] - helpful_embeddings[:, 1]
            q_true, _ = np.linalg.qr(true_vectors.T)
            q_true = q_true.T

            for orthonormal_vector in q_true:
                sim = image_embeddings @ orthonormal_vector
                helpful_features = sim[:, None] * np.repeat(orthonormal_vector[None, :], sim.shape[0], axis=0) / np.linalg.norm(orthonormal_vector)
                image_embeddings = image_embeddings + helpful_features
                image_embeddings = image_embeddings / np.linalg.norm(image_embeddings, axis=1, keepdims=True)

        return torch.from_numpy(image_embeddings).float().to(device)
