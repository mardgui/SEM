import torch.nn.functional as F

from .method_utils import *


class OrthProj(CLIPEmbeddingDebiasingMethod):
    def __init__(self, config):
        super().__init__(config)
        self.normalize = config['normalize']

    def debias_text(self, text_prompts, *args, **kwargs):
        extra_prompts = kwargs['extra_prompts']
        device = self.model.text_projection.device

        with torch.no_grad():
            query_embedding = get_clip_text_embedding(text_prompts, self.model, normalize=self.normalize).cpu().float().numpy().astype(np.float64)
            spurious_embeddings = get_clip_text_embedding(extra_prompts['spurious prompts'], self.model, normalize=self.normalize).cpu().float().numpy().astype(np.float64)

        P0 = get_proj_matrix(spurious_embeddings)

        debiased = torch.from_numpy(query_embedding @ P0).float().to(device)
        if self.normalize:
            debiased = F.normalize(debiased, dim=-1)
        
        return debiased

    def debias_images(self, image_embeddings, *args, **kwargs):
        return image_embeddings.float()


def get_A(z_i, z_j):
    z_i = z_i[:, None]
    z_j = z_j[:, None]
    return z_i @ z_i.T + z_j @ z_j.T - z_i @ z_j.T - z_j @ z_i.T


def get_M(embeddings, S):
    d = embeddings.shape[1]
    M = np.zeros((d, d), dtype=np.float64)
    for s in S:
        M  += get_A(embeddings[s[0]], embeddings[s[1]])
    return M / len(S)


class OrthCali(CLIPEmbeddingDebiasingMethod):
    def __init__(self, config):
        super().__init__(config)
        self.lam = config['lam']
        self.normalize = config['normalize']

    def debias_text(self, text_prompts, *args, **kwargs):
        extra_prompts = kwargs['extra_prompts']
        device = self.model.text_projection.device

        with torch.no_grad():
            query_embedding = get_clip_text_embedding(text_prompts, self.model, self.normalize).cpu().float().numpy().astype(np.float64)
            spurious_embeddings = get_clip_text_embedding(extra_prompts['spurious prompts'], self.model, self.normalize).cpu().float().numpy().astype(np.float64)
            candidate_embeddings = get_clip_text_embedding(extra_prompts['candidate prompts'], self.model, self.normalize).cpu().float().numpy().astype(np.float64)
        try:
            S = extra_prompts['S']
        except KeyError:
            S = []
            m = query_embedding.shape[0]
            n = spurious_embeddings.shape[0]
            for i in range(m):
                    for j in range(n):
                        for k in range(j + 1, n):
                            S.append([i * n + j, i * n + k])

        P0 = get_proj_matrix(spurious_embeddings)
        M = get_M(candidate_embeddings, S)
        G = self.lam * M + np.eye(M.shape[0], dtype=np.float64)
        P = P0 @ np.linalg.inv(G)
        
        debiased = torch.from_numpy(query_embedding @ P.T).float().to(device)
        if self.normalize:
            debiased = F.normalize(debiased, dim=-1)
        
        return debiased

    def debias_images(self, image_embeddings, *args, **kwargs):
        return image_embeddings.float()
