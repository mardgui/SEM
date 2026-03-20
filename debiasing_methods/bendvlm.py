import torch.nn.functional as F
from scipy.optimize import minimize

from .method_utils import *


def get_cos_neighbors(query_embedding, dataset_embeddings, k):
    query_embedding = F.normalize(query_embedding, dim=-1)
    dataset_embeddings = F.normalize(dataset_embeddings, dim=-1)
    cos_scores = (query_embedding @ dataset_embeddings.T).mean(dim=0)
    top_results = torch.topk(cos_scores, k=k)
    return top_results.values.cpu().numpy(), top_results.indices.cpu().numpy()


def norm_constraint(e_star):
    return (np.dot(e_star, e_star) - 1).astype(np.float64)


def eq_dist_constraint(e_star, y_mean, x_mean):
    return np.dot(e_star, y_mean) - np.dot(e_star, x_mean).astype(np.float64)


def objective(e_star, initial_e):
    return -np.dot(e_star, initial_e).astype(np.float64)


class BendVLM(CLIPEmbeddingDebiasingMethod):
    def __init__(self, config):
        super().__init__(config)
        self.normalize = config['normalize']
        self.num_neighbors = config['num_neighbors']

    def debias_text(self, text_prompts, *args, **kwargs):
        with torch.no_grad():
            device = self.model.text_projection.device
            extra_prompts = kwargs['extra_prompts']

            query_embedding = get_clip_text_embedding(text_prompts, self.model, self.normalize).cpu().float().numpy().astype(np.float64)
            spurious_embeddings = get_clip_text_embedding(extra_prompts['spurious prompts'], self.model, self.normalize).cpu().float().numpy().astype(np.float64)
            inclusive_candidate_embeddings = get_clip_text_embedding(extra_prompts['augmentations'], self.model, self.normalize).cpu().float().numpy().astype(np.float64)
            
            S = []
            m = query_embedding.shape[0]
            n = spurious_embeddings.shape[0]
            for i in range(m):
                for j in range(n):
                    for k in range(j + 1, n):
                        S.append([i * n + j, i * n + k])

            rewrite_pair_list = []
            for e_i, e_j in S:
                rewrite_pair_list.append(((inclusive_candidate_embeddings[e_i] - inclusive_candidate_embeddings[e_j]) / 2))

            sub_local_embeddings = np.stack(rewrite_pair_list, axis=0)
            sub_local_embeddings = np.concatenate([spurious_embeddings, sub_local_embeddings], axis=0)

            P0_local = get_proj_matrix(sub_local_embeddings)
            P0_local_embeddings = torch.from_numpy(query_embedding @ P0_local).float().to(device)
            if self.normalize:
                P0_local_embeddings =  F.normalize(P0_local_embeddings, dim=-1)

            reference_embeddings = kwargs['reference_embeddings'].float()
            reference_metadata = kwargs['reference_metadata']
            bias_attribute = kwargs['bias_attribute']
            bias_classes = kwargs['bias_classes']

            reference_bias_labels = reference_metadata[bias_attribute].values
            
            anchor_bias_class = bias_classes[0]
            anchor_reference_embeddings = reference_embeddings[reference_bias_labels == anchor_bias_class]

            _, topk_indices = get_cos_neighbors(P0_local_embeddings, anchor_reference_embeddings, k=self.num_neighbors)

            anchor_reference_embeddings = anchor_reference_embeddings[topk_indices].cpu().numpy().astype(np.float64)
            anchor_prototype = anchor_reference_embeddings.mean(axis=0)

            bc_prototypes = []

            for bias_class in bias_classes[1:]:
                bc_reference_embeddings_dataset = reference_embeddings[reference_bias_labels == bias_class]
                _, topk_indices = get_cos_neighbors(P0_local_embeddings, bc_reference_embeddings_dataset, k=self.num_neighbors)
                bc_reference_embeddings = bc_reference_embeddings_dataset[topk_indices].cpu().numpy().astype(np.float64)
                bc_prototype = bc_reference_embeddings.mean(axis=0)
                bc_prototypes.append(bc_prototype)
                
            norm_con = {'type': 'eq', 'fun': norm_constraint}
            cons = [norm_con]
            for bc_prototype in bc_prototypes:
                dist_con = {'type': 'eq', 'fun': eq_dist_constraint, 'args': (anchor_prototype, bc_prototype)}
                cons.append(dist_con)

            solutions = []
            for x0 in P0_local_embeddings:
                x0 = x0.cpu().numpy().astype(np.float64)
                solutions.append(minimize(objective, x0, args=(x0,), method='SLSQP', constraints=cons).x)
            debiased = torch.from_numpy(np.stack(solutions, axis=0)).float().to(device)
            if self.normalize:
                debiased = F.normalize(debiased, dim=-1)
            return debiased

    def debias_images(self, image_embeddings, *args, **kwargs):
        return image_embeddings.float()
