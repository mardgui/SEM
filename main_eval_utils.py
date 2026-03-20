import os

import numpy as np
import pandas as pd
import scipy.stats as st
import torch
import torch.nn.functional as F

from debiasing_methods.bendvlm import BendVLM
from debiasing_methods.bendsem import BendSEM
from debiasing_methods.biased_prompts import OrthCali, OrthProj
from debiasing_methods.prism_inference import PRISM, PRISM_mini
from debiasing_methods.roboshot import RoboShot
from debiasing_methods.sem import SEM
from debiasing_methods.vanilla_clip import VanillaCLIP
from debiasing_methods.zsdebias_inference import ZSDebias
    

class EmbeddingDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path, model_ID, split, normalize):
        self.meta_df = pd.read_feather(os.path.join(dataset_path, 'metadata.feather'))
        if split == 'train':
            self.indices = np.load(os.path.join(dataset_path, 'train_indices.npy'))
        elif split == 'val':
            self.indices = np.load(os.path.join(dataset_path, 'val_indices.npy'))
        else:
            raise ValueError('Please select either \'train\' or \'val\' split.')
        self.embeddings = np.load(os.path.join(dataset_path, f'CLIP_{model_ID}_embeddings.npy'))
        self.split = split
        self.normalize = normalize
        self.fold = None

    def set_fold(self, k):
        self.fold = k

    def __len__(self):
        assert self.fold is not None, 'Please select a fold first.'
        return len(self.indices[self.fold])

    def __getitem__(self, index):
        assert self.fold is not None, 'Please select a fold first.'
        metadata = self.meta_df.iloc[self.indices[self.fold]].iloc[:, index]
        embedding = self.embeddings[self.indices[self.fold]][index]
        if self.normalize:
            embedding /= np.linalg.norm(embedding, axis=-1, keepdims=True)
        return metadata, embedding
    

class EmbeddingClassificationDataset(EmbeddingDataset):
    def __init__(self, dataset_path, model_ID, split, target_attribute, target_dict, bias_attribute, bias_dict, normalize):
        super().__init__(dataset_path, model_ID, split, normalize)
        self.target_attribute = target_attribute
        self.target_dict = target_dict
        self.bias_attribute = bias_attribute
        self.bias_dict = bias_dict
    
        target_labels = [target_dict[v] for v in self.meta_df[target_attribute].values]
        bias_labels = [bias_dict[v] for v in self.meta_df[bias_attribute].values]
        self.meta_df = pd.DataFrame({target_attribute: target_labels, bias_attribute: bias_labels})


def get_cos_neighbors(query_embedding, dataset_embeddings, k):
    query_embedding = F.normalize(query_embedding, dim=-1)
    dataset_embeddings = F.normalize(dataset_embeddings, dim=-1)
    cos_scores = (query_embedding @ dataset_embeddings.T).squeeze()
    top_results = torch.topk(cos_scores, k=k)
    return top_results.values.cpu().numpy(), top_results.indices.cpu().numpy()


def log_with_eps(x, eps=1e-10):
    if x < eps:
        return np.log(eps)
    else:
        return np.log(x)


def kl(retrieved_metadata, bias_prior, bias_attribute, bias_classes):
    kl = 0
    for c in bias_classes:
        p_y_ds = bias_prior[c]
        p_y_returned = (retrieved_metadata[bias_attribute].values == c).mean()
        kl += p_y_returned * log_with_eps(p_y_returned / p_y_ds)
    return kl


def max_skew(retrieved_metadata, bias_prior, bias_attribute, bias_classes):
    maxskew = 0
    for c in bias_classes:
        p_y_ds = bias_prior[c]
        p_y_returned = (retrieved_metadata[bias_attribute].values == c).mean()
        candidate_skew = log_with_eps(p_y_returned / p_y_ds)
        if candidate_skew > maxskew:
            maxskew = candidate_skew
    return maxskew


def mean_confidence_interval(data, confidence=0.95):
    return st.sem(data) * st.t.ppf((1 + confidence) / 2., len(data) - 1)


def initialize_method(method_name, config):
    method_dict = {
        'vanilla': VanillaCLIP,
        'orth_proj': OrthProj,
        'orth_cali': OrthCali,
        'bendvlm': BendVLM,
        'bendsem': BendSEM,
        'prism': PRISM,
        'prism_mini': PRISM_mini,
        'roboshot': RoboShot,
        'sem': SEM,
        'zsdebias': ZSDebias
    }
    try:
        return method_dict[method_name](config)
    except KeyError:
        raise ValueError(f'Unknown method: {method_name}')
