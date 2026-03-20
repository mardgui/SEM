import argparse
import json
import os

import clip
import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm.auto import tqdm


parser = argparse.ArgumentParser()
parser.add_argument('--model_id', type=str, default='ViT-L/14@336px')
parser.add_argument('--batch_size', type=int, default=512)
parser.add_argument('--dataset_path', type=str, default='./data/CelebA')
parser.add_argument('--dataset_name', type=str, default='celeba')
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_ID = args.model_id
batch_size = args.batch_size
dataset_path = args.dataset_path
dataset_name = args.dataset_name

model, preprocess = clip.load(model_ID, device=device)
model_ID = model_ID.replace('/', '-')

output_file = os.path.join(dataset_path, f'CLIP_{model_ID}_embeddings.npy')


def get_celeba_metadata(dataset_path):
    celeba_df = pd.read_csv(os.path.join(dataset_path, 'Anno', 'list_attr_celeba.txt'), skiprows=1, sep='\s+')
    celeba_df['gender'] = ['Male' if c == 1 else 'Female' for c in celeba_df['Male'].values]
    celeba_df = celeba_df.reset_index()
    celeba_df = celeba_df.rename(columns = {'index': 'file_name'})
    celeba_df['file_path'] = [os.path.join('Img', 'img_align_celeba', f) for f in celeba_df['file_name']]
    return celeba_df


def get_fairface_metadata(dataset_path):
    fairface_train_df = pd.read_csv(os.path.join(dataset_path, 'fairface_label_train.csv'))
    fairface_val_df = pd.read_csv(os.path.join(dataset_path, 'fairface_label_val.csv'))
    fairface_df = pd.concat([fairface_train_df, fairface_val_df]).reset_index(drop=True)
    fairface_df = fairface_df.rename(columns={'file': 'file_path'})
    fairface_df['file_name'] = [os.path.basename(p) for p in fairface_df['file_path']]
    return fairface_df


def get_utkface_metadata(dataset_path):
    utk_gender_map = {
        '0': 'Male',
        '1': 'Female'
    }
    utk_race_map = {
        '0': 'White',
        '1': 'Black',
        '2': 'Asian',
        '3': 'Indian',
        '4': 'Latino Hispanic'
    }

    image_files = os.listdir(os.path.join(dataset_path, 'images'))
    image_records = []

    for image_file in image_files:
        try:
            age, gender, race, _ = image_file.split('_')
        except ValueError:
            continue
        image_records.append(
            {'file_name': image_file,
             'age': int(age),
             'gender': utk_gender_map[gender],
             'race': utk_race_map[race],
             'file_path': os.path.join('images', image_file)}
        )

    return pd.DataFrame.from_records(image_records)


def get_waterbirds_metadata(dataset_path):
    waterbirds_map = {
        0: 'land bird',
        1: 'water bird'
    }
    waterbirds_background_map = {
        0: 'land background',
        1: 'water background'
    }
    
    waterbirds_df = pd.read_csv(os.path.join(dataset_path, 'waterbird_complete95_forest2water2', 'metadata.csv'))
    waterbirds_df = waterbirds_df.drop(columns=['img_id', 'place_filename']).rename(columns={'img_filename': 'file_path', 'y': 'bird type', 'place': 'background'})
    waterbirds_df['file_name'] = [os.path.basename(p) for p in waterbirds_df['file_path']]
    waterbirds_df['file_path'] = [os.path.join('waterbird_complete95_forest2water2', p) for p in waterbirds_df['file_path']]
    waterbirds_df['bird type'] = [waterbirds_map[y] for y in waterbirds_df['bird type']]
    waterbirds_df['background'] = [waterbirds_background_map[b] for b in waterbirds_df['background']]
    return waterbirds_df


def get_coco_metadata(dataset_path):
    with open(os.path.join(dataset_path, 'instances_train2017.json'), 'r') as f:
        coco_data = json.load(f)

    animal_cat_ids = set([cat['id'] for cat in coco_data['categories'] if cat['supercategory'] == 'animal'])
    
    valid_img_ids = set()
    for ann in coco_data['annotations']:
        if ann['category_id'] in animal_cat_ids:
            valid_img_ids.add(ann['image_id'])
            
    coco_records = []
    for img in coco_data['images']:
        if img['id'] in valid_img_ids:
            coco_records.append({
                'file_name': img['file_name'],
                'file_path': os.path.join('train2017', img['file_name'])
            })
            
    return pd.DataFrame.from_records(coco_records)


def get_imagenet100_metadata(dataset_path):
    with open(os.path.join(dataset_path, 'imagenet100.txt'), 'r') as f:
        target_classes = [line.strip() for line in f.readlines() if line.strip()]

    image_records = []
    train_dir = os.path.join(dataset_path, 'train')
    
    for class_id in tqdm(target_classes):
        class_dir = os.path.join(train_dir, class_id)
        if not os.path.isdir(class_dir):
            continue
            
        for file in os.listdir(class_dir):
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                rel_path = os.path.join('train', class_id, file)
                image_records.append({
                    'file_name': file,
                    'file_path': rel_path,
                    'class_id': class_id
                })
                
    return pd.DataFrame.from_records(image_records)


def get_lfw_metadata(dataset_path):
    image_records = []

    for person_name in sorted(os.listdir(os.path.join(dataset_path, 'images'))):
        person_dir = os.path.join(dataset_path, 'images', person_name)
        for image_file in sorted(os.listdir(person_dir)):
            if image_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_records.append({'file_name': image_file,
                                      'file_path': os.path.join('images', person_name, image_file)})
                
    return pd.DataFrame.from_records(image_records)


def get_facet_metadata(dataset_path):
    facet_df = pd.read_csv(os.path.join(dataset_path, 'annotations.csv'))
    facet_df = facet_df[(facet_df['visible_torso'] == 1) & (facet_df['visible_face'] == 1)]
    facet_df['file_path'] = [os.path.join('images', f) for f in facet_df['filename']]
    facet_df['file_name'] = [os.path.basename(p) for p in facet_df['file_path']]
    return facet_df


if os.path.exists(os.path.join(dataset_path, f'metadata.feather')):
    print(f"Loading existing metadata from {dataset_path}...")
    meta_df = pd.read_feather(os.path.join(dataset_path, f'metadata.feather'))
else:
    print(f"Generating metadata for {dataset_name}...")
    if dataset_name == 'celeba':
        meta_df = get_celeba_metadata(dataset_path)
    elif dataset_name == 'coco':
        meta_df = get_coco_metadata(dataset_path)
    elif dataset_name == 'facet':
        meta_df = get_facet_metadata(dataset_path)
    elif dataset_name == 'fairface':
        meta_df = get_fairface_metadata(dataset_path)
    elif dataset_name == 'imagenet100':
        meta_df = get_imagenet100_metadata(dataset_path)
    elif dataset_name == 'lfw':
        meta_df = get_lfw_metadata(dataset_path)
    elif dataset_name == 'utkface':
        meta_df = get_utkface_metadata(dataset_path)
    elif dataset_name == 'waterbirds':
        meta_df = get_waterbirds_metadata(dataset_path)
    else:
        print(f'Dataset "{dataset_name}" is not part of the available datasets. Available datasets: ["celeba", "coco", "fairface", "imagenet100", "utkface", "waterbirds"]')
        exit(1)
    meta_df.to_feather(os.path.join(dataset_path, f'metadata.feather'))

if os.path.exists(output_file):
    print(f'{output_file} already exists. Delete the file manually if you wish to compute the embeddings again.')
    exit(0)

print(f"Computing embeddings for {len(meta_df)} images...")
BS = min(batch_size, len(meta_df))
iterations = len(meta_df) // BS + ((len(meta_df) % BS) != 0)
embeddings = []

for i in tqdm(range(iterations)):
    batch_files = meta_df.iloc[i * BS:(i + 1) * BS]['file_path'].values
    batch_imgs = [Image.open(os.path.join(dataset_path, f)) for f in batch_files]

    with torch.no_grad():
        image_inputs = torch.stack([preprocess(img) for img in batch_imgs]).to(device)
        image_features = model.encode_image(image_inputs).to('cpu').numpy()
    
    embeddings.append(image_features)

embeddings = np.concatenate(embeddings, axis=0)
np.save(output_file, embeddings)
print(f"Saved {embeddings.shape} to {output_file}")
