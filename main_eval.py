import argparse
import json
import os

import clip
import numpy as np
import pandas as pd
import torch
import yaml
from tqdm.auto import tqdm

import classes_and_prompts
from main_eval_utils import *


parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='config/waterbirds_zs-classification_waterbirds_clip-RN50_sem-b.yml')
args = parser.parse_args()

print(args.config)
with open(args.config, 'r') as f:
    config = yaml.safe_load(f)

dataset_name = config['dataset_name']
dataset_path = config['dataset_path']
task = config['task']
if task == 'retrieval':
    K = config['retrieval_K']
    query_type = config['query_type']
    bias_attribute = config['bias_attribute']
if task == 'zs_classification':
    class_setting = config['class_setting']
model_ID = config['model_ID']
method = config['method']
normalize = config['normalize']

np.random.seed(0)
torch.manual_seed(0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model, preprocess = clip.load(model_ID, device=device)
model_ID = model_ID.replace('/', '-')

if task == 'retrieval':
    try:
        query_classes = classes_and_prompts.retrieval_classes[query_type]['classes']
        query_is_labeled = classes_and_prompts.retrieval_classes[query_type]['is_labeled']
    except KeyError:
        print(f'Task {task} does not support the "{query_type}" query type.')
        exit(1)

    _bias_attribute = bias_attribute
    if bias_attribute == 'race':
        if dataset_name == 'utkface':
            bias_classes = ['White', 'Black', 'Asian', 'Indian', 'Latino Hispanic']
            _bias_attribute = 'utkrace'
        elif dataset_name == 'fairface':
            bias_classes = ['Black', 'East Asian', 'Indian', 'Latino_Hispanic', 'Middle Eastern', 'Southeast Asian', 'White']
        else:
            print(f'Dataset {dataset_name} does not support the "race" bias attribute.')
            exit(1)
    elif bias_attribute == 'gender':
        assert dataset_name in ['celeba', 'fairface', 'utkface'], f'Dataset {dataset_name} does not support the "gender" bias attribute.'
        bias_classes = ['Male', 'Female']
    else:
        print(f'Unknown or unsupported bias attribute: {bias_attribute}. Bias attribute must be either "race" or "gender"')
        exit(1)

    with open(f'query_templates/{query_type}_{_bias_attribute}_query_templates.json', 'r') as file:
        query_dict = json.load(file)
    
    train_dataset = EmbeddingDataset(dataset_path, model_ID, 'train', normalize)
    val_dataset = EmbeddingDataset(dataset_path, model_ID, 'val', normalize)

elif task == 'zs_classification':
    try:
        class_prompts = classes_and_prompts.zero_shot_prompts[class_setting]['prompts']
        target_attribute = classes_and_prompts.zero_shot_prompts[class_setting]['target_attribute']
        target_dict = classes_and_prompts.zero_shot_prompts[class_setting]['target_dict']
        bias_attribute = classes_and_prompts.zero_shot_prompts[class_setting]['bias_attribute']
        bias_dict = classes_and_prompts.zero_shot_prompts[class_setting]['bias_dict']
        bias_classes = list(bias_dict.values())
    except KeyError:
        print(f'Task {task} does not support the "{class_setting}" classification setting.')
        exit(1)

    with open(f'prompt_templates/{class_setting}_prompt_templates.json', 'r') as file:
        prompt_dict = json.load(file)
        
    train_dataset = EmbeddingClassificationDataset(dataset_path, model_ID, 'train',
                                                   target_attribute, target_dict,
                                                   bias_attribute, bias_dict,
                                                   normalize)
    val_dataset = EmbeddingClassificationDataset(dataset_path, model_ID, 'val',
                                                 target_attribute, target_dict,
                                                 bias_attribute, bias_dict,
                                                 normalize)
else:
    print(f'Unknown or unsupported task: {task}. Task must be either "retrieval" or "zs_classification"')
    exit(1)

k_folds = train_dataset.indices.shape[0]
assert k_folds == val_dataset.indices.shape[0]
config['k_folds'] = k_folds

if task == 'retrieval':
    task_full = f'retrieval-{query_type}'
else:
    task_full = task

if task == 'retrieval':
    config['query_classes'] = query_classes
    config['bias_attribute'] = _bias_attribute
config['clip_model'] = model
method_name = method
method = initialize_method(method, config)

all_folds_df = []

with torch.no_grad():
    for k_fold in tqdm(range(k_folds)):
        train_dataset.set_fold(k_fold)
        val_dataset.set_fold(k_fold)
        
        eval_metadata, eval_embeddings = val_dataset[:]
        eval_embeddings = torch.from_numpy(eval_embeddings).to(device)

        reference_metadata, reference_embeddings = train_dataset[:]
        reference_embeddings = torch.from_numpy(reference_embeddings).to(device)

        if task == 'retrieval':
            result_records = []

            for query_class in query_classes:
                query_prompt = [query_dict[query_class]['query']]
                extra_prompts = query_dict[query_class]['extra prompts'][method_name]

                bias_prior = {}
                for bias_class in bias_classes:
                    bias_prior[bias_class] = (eval_metadata[bias_attribute].values == bias_class).mean()

                d_text_embeddings = method.debias_text(query_prompt,
                                                       extra_prompts=extra_prompts,
                                                       reference_embeddings=reference_embeddings,
                                                       reference_metadata=reference_metadata,
                                                       bias_attribute=bias_attribute,
                                                       bias_classes=bias_classes,
                                                       query_class=query_class)
            
                d_image_embeddings = method.debias_images(eval_embeddings,
                                                          extra_prompts=extra_prompts,
                                                          query_class=query_class)

                _, topk_indices = get_cos_neighbors(d_text_embeddings, d_image_embeddings, k=K)
                retrieved_embeddings = d_image_embeddings[topk_indices].cpu().numpy()
                
                retrieved_metadata = eval_metadata.iloc[topk_indices]

                record = {'k_fold': k_fold, 'query_class': query_class}
                
                record['kl'] = kl(retrieved_metadata, bias_prior, bias_attribute, bias_classes)
                record['max_skew'] = max_skew(retrieved_metadata, bias_prior, bias_attribute, bias_classes)
                
                if query_is_labeled:
                    record['precision'] = (retrieved_metadata[query_class] == 1).mean()

                result_records.append(record)
            
            all_folds_df.append(pd.DataFrame.from_records(result_records))
            
        elif task == 'zs_classification':
            extra_prompts = prompt_dict['extra prompts'][method_name]

            d_text_embeddings = F.normalize(method.debias_text(class_prompts,
                                                               extra_prompts=extra_prompts,
                                                               reference_embeddings=reference_embeddings,
                                                               reference_metadata=reference_metadata,
                                                               bias_attribute=bias_attribute,
                                                               bias_classes=bias_classes), dim=-1)
            
            d_image_embeddings = F.normalize(method.debias_images(eval_embeddings,
                                                                  extra_prompts=extra_prompts), dim=-1).to(d_text_embeddings.dtype)

            preds = (d_image_embeddings @ d_text_embeddings.T).argmax(dim=1).cpu().numpy()

            target_labels = eval_metadata[target_attribute].values
            bias_labels = eval_metadata[bias_attribute].values

            correct = preds == target_labels

            fold_df = pd.DataFrame({
                'k_fold': k_fold, 'target_label': target_labels,
                'pred': preds, 'correct': correct, 'bias_label': bias_labels
            })
            all_folds_df.append(fold_df)

df = pd.concat(all_folds_df).reset_index(drop=True)

result_records = []
result_base_dict = {'dataset': dataset_name, 'task': task, 'backbone': model_ID, 'method': method_name, 'normalize': normalize}
if task == 'retrieval':
    result_base_dict['setting'] = f'query={query_type}|bias={bias_attribute}|K={K}'
elif task == 'zs_classification':
    result_base_dict['setting'] = class_setting

if task == 'retrieval':
    metrics = df.columns.values[2:]
    for metric in metrics:
        mean_metric = df[metric].values.mean()
        metric_ci = mean_confidence_interval(df[metric].values)
        print(f'{metric}: {mean_metric:.3f} +- {metric_ci:.3f}')
        result_records.append({**result_base_dict, 'metric': metric, 'mean': mean_metric, 'ci': metric_ci})
elif task == 'zs_classification':
    mean_accuracy = df['correct'].values.mean()
    accuracy_ci = mean_confidence_interval(df['correct'].values)
    print(f'Mean accuracy: {mean_accuracy:.3f} +- {accuracy_ci:.3f}')
    result_records.append({**result_base_dict, 'metric': 'Acc.', 'mean': mean_accuracy, 'ci': accuracy_ci})
    grouped_df = df.drop(columns=['k_fold', 'pred']).groupby(['target_label', 'bias_label']).mean()
    worst_group_accuracy = grouped_df['correct'].min()
    worst_target_label, worst_bias_label = grouped_df.index[grouped_df['correct'].argmin()]
    worst_group_accuracy_ci = mean_confidence_interval(df[(df['target_label'] == worst_target_label) & (df['bias_label'] == worst_bias_label)]['correct'].values)
    print(f'Worst-group accuracy: {worst_group_accuracy:.3f} +- {worst_group_accuracy_ci:.3f}')
    result_records.append({**result_base_dict, 'metric': 'WG Acc.', 'mean': worst_group_accuracy, 'ci': worst_group_accuracy_ci})

os.makedirs('main_eval_output', exist_ok=True)
output_file_name = f'main_eval_output/{args.config.split("/")[-1][:-4]}.json'
with open(output_file_name, 'w') as f:
    json.dump(result_records, f)
