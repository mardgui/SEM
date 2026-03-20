import clip
import numpy as np
import torch
from sklearn.decomposition import TruncatedSVD


class CLIPEmbeddingDebiasingMethod:
    def __init__(self, config):
        self.model = config['clip_model']

    def debias_text(self, text_prompts, *args, **kwargs):
        raise NotImplementedError

    def debias_images(self, image_embeddings, *args, **kwargs):
        raise NotImplementedError


def get_clip_text_embedding(text_prompts, model, normalize):
    with torch.no_grad():
        text_tokens = clip.tokenize(text_prompts).to(model.text_projection.device)
        text_embeddings = model.encode_text(text_tokens)
        if normalize:
            text_embeddings /= text_embeddings.norm(dim=-1, keepdim=True)
    return text_embeddings


def get_proj_matrix(embeddings):
    tSVD = TruncatedSVD(n_components=len(embeddings))
    tSVD.fit(embeddings)
    basis = tSVD.components_.T.astype(np.float64)

    proj = basis @ basis.T
    proj = np.eye(proj.shape[0], dtype=np.float64) - proj
    return proj
