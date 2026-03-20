from .bendvlm import *
from .sem import *


class BendSEM(SEM):
    def __init__(self, config):
        super().__init__(config)
        self.normalize = config['normalize']
        self.num_neighbors = config['num_neighbors']

    def debias_text(self, text_prompts, *args, **kwargs):
        with torch.no_grad():
            P0_local_embeddings = super().debias_text(text_prompts, *args, **kwargs)

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
            debiased = torch.from_numpy(np.stack(solutions, axis=0)).float().to(self.device)
            if self.normalize:
                debiased = F.normalize(debiased, dim=-1)
            return debiased
