import torch.nn.functional as F

from .method_utils import *
from .prism_training import get_transformer


retrieval_setting_dict = {
    ('hair', 'gender'): 'celeba_<qc>_gender',
    ('stereotype', 'gender'): 'stereotype_<qc>_gender',
    ('stereotype', 'race'): 'stereotype_<qc>_race',
    ('stereotype', 'utkrace'): 'stereotype_<qc>_utkrace'
}


class PRISM(CLIPEmbeddingDebiasingMethod):
    def __init__(self, config):
        super().__init__(config)
        device = self.model.text_projection.device
        saved_model_file_prefix = f'prism_clip={config["model_ID"].replace("/", "-")}_task={config["task"]}'
        saved_model_file_suffix = f'epochs={config["epochs"]}_num_bases={config["num_bases"]}_num_samples={config["num_samples"]}'
        saved_model_file_suffix += f'_seed={config["seed"]}_lr={config["lr"]}_batch_size={config["batch_size"]}'
        saved_model_file_suffix += f'_wd={config["wd"]}_init_weight={config["init_weight"]}.pth'
        if config['task'] == 'retrieval':
            self.transformer = {}
            for query_class in config['query_classes']:
                self.transformer[query_class] = get_transformer(config['model_ID'], config['num_bases'], config['init_weight'], device)
                task_setting = retrieval_setting_dict[(config["query_type"], config['bias_attribute'])].replace('<qc>', query_class.replace(' ', '-')).lower()
                saved_model_file = f'{saved_model_file_prefix}_setting={task_setting}_{saved_model_file_suffix}'
                try:
                    self.transformer[query_class].load_state_dict(torch.load(f'debiasing_methods/prism_weights/{saved_model_file}',
                                                                             map_location=device,
                                                                             weights_only=True))
                    self.transformer[query_class].eval()
                except FileNotFoundError:
                    print(f'{saved_model_file} does not exist. Please run debiasing_methods/prism_training.py first.')
                    exit(1)
        elif config['task'] == 'zs_classification':
            self.transformer = get_transformer(config['model_ID'], config['num_bases'], config['init_weight'], device)
            task_setting = config["class_setting"]
            saved_model_file = f'{saved_model_file_prefix}_setting={task_setting}_{saved_model_file_suffix}'
            try:
                self.transformer.load_state_dict(torch.load(f'debiasing_methods/prism_weights/{saved_model_file}',
                                                            map_location=device,
                                                            weights_only=True))
                self.transformer.eval()
            except FileNotFoundError:
                print(f'{saved_model_file} does not exist. Please run debiasing_methods/prism_training.py first.')
                exit(1)
        

    def debias_text(self, text_prompts, *args, **kwargs):
        if 'query_class' in kwargs:
            transformer = self.transformer[kwargs['query_class']]
        else:
            transformer = self.transformer
        
        with torch.no_grad():
            query_embedding = get_clip_text_embedding(text_prompts, self.model, normalize=True).float()
            debiased = transformer(query_embedding)
        
        return debiased

    def debias_images(self, image_embeddings, *args, **kwargs):
        if 'query_class' in kwargs:
            transformer = self.transformer[kwargs['query_class']]
        else:
            transformer = self.transformer
        
        with torch.no_grad():
            debiased = transformer(image_embeddings.float())
        
        return debiased


def get_prism_proj_matrix(spurious_embeddings):
    V = spurious_embeddings.T
    VtV = V.T @ V
    VtV_inv = torch.inverse(VtV)
    P = torch.eye(V.shape[0], device=V.device, dtype=torch.float32) - V @ VtV_inv @ V.T
    return P


class PRISM_mini(CLIPEmbeddingDebiasingMethod):
    def __init__(self, config):
        super().__init__(config)

    def debias_text(self, text_prompts, *args, **kwargs):
        extra_prompts = kwargs['extra_prompts']

        with torch.no_grad():
            query_embedding = get_clip_text_embedding(text_prompts, self.model, normalize=True).float()
            spurious_embeddings = get_clip_text_embedding(extra_prompts['spurious prompts'], self.model, normalize=True).float()
        
        P = get_prism_proj_matrix(spurious_embeddings)
        debiased = query_embedding @ P

        return debiased

    def debias_images(self, image_embeddings, *args, **kwargs):
        extra_prompts = kwargs['extra_prompts']

        with torch.no_grad():
            spurious_embeddings = get_clip_text_embedding(extra_prompts['spurious prompts'], self.model, normalize=True).float()
            
        P = get_prism_proj_matrix(spurious_embeddings)
        debiased = F.normalize(image_embeddings.float(), dim=-1) @ P

        return debiased
