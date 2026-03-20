# Embeddings mandatory for evaluation

## ViT-B/16
python precompute_embeddings.py --model_id "ViT-B/16" --batch_size 512 --dataset_path "./data/CelebA" --dataset_name "celeba"
python precompute_embeddings.py --model_id "ViT-B/16" --batch_size 512 --dataset_path "./data/FairFace" --dataset_name "fairface"
python precompute_embeddings.py --model_id "ViT-B/16" --batch_size 512 --dataset_path "./data/UTKFace" --dataset_name "utkface"
python precompute_embeddings.py --model_id "ViT-B/16" --batch_size 512 --dataset_path "./data/Waterbirds" --dataset_name "waterbirds"

## ViT-L/14@336px
python precompute_embeddings.py --model_id "ViT-L/14@336px" --batch_size 512 --dataset_path "./data/CelebA" --dataset_name "celeba"
python precompute_embeddings.py --model_id "ViT-L/14@336px" --batch_size 512 --dataset_path "./data/FairFace" --dataset_name "fairface"
python precompute_embeddings.py --model_id "ViT-L/14@336px" --batch_size 512 --dataset_path "./data/UTKFace" --dataset_name "utkface"
python precompute_embeddings.py --model_id "ViT-L/14@336px" --batch_size 512 --dataset_path "./data/Waterbirds" --dataset_name "waterbirds"

## ResNet-50
python precompute_embeddings.py --model_id "RN50" --batch_size 512 --dataset_path "./data/CelebA" --dataset_name "celeba"
python precompute_embeddings.py --model_id "RN50" --batch_size 512 --dataset_path "./data/FairFace" --dataset_name "fairface"
python precompute_embeddings.py --model_id "RN50" --batch_size 512 --dataset_path "./data/UTKFace" --dataset_name "utkface"
python precompute_embeddings.py --model_id "RN50" --batch_size 512 --dataset_path "./data/Waterbirds" --dataset_name "waterbirds"

## ResNet-101
python precompute_embeddings.py --model_id "RN101" --batch_size 512 --dataset_path "./data/CelebA" --dataset_name "celeba"
python precompute_embeddings.py --model_id "RN101" --batch_size 512 --dataset_path "./data/FairFace" --dataset_name "fairface"
python precompute_embeddings.py --model_id "RN101" --batch_size 512 --dataset_path "./data/UTKFace" --dataset_name "utkface"
python precompute_embeddings.py --model_id "RN101" --batch_size 512 --dataset_path "./data/Waterbirds" --dataset_name "waterbirds"


# Embeddings needed for ZSDebias training only

## ViT-B/16
python precompute_embeddings.py --model_id "ViT-B/16" --batch_size 512 --dataset_path "./data/COCO" --dataset_name "coco"
python precompute_embeddings.py --model_id "ViT-B/16" --batch_size 512 --dataset_path "./data/FACET" --dataset_name "facet"
python precompute_embeddings.py --model_id "ViT-B/16" --batch_size 512 --dataset_path "./data/ImageNet-100" --dataset_name "imagenet100"
python precompute_embeddings.py --model_id "ViT-B/16" --batch_size 512 --dataset_path "./data/LFW" --dataset_name "lfw"

## ViT-L/14@336px
python precompute_embeddings.py --model_id "ViT-L/14@336px" --batch_size 512 --dataset_path "./data/COCO" --dataset_name "coco"
python precompute_embeddings.py --model_id "ViT-L/14@336px" --batch_size 512 --dataset_path "./data/FACET" --dataset_name "facet"
python precompute_embeddings.py --model_id "ViT-L/14@336px" --batch_size 512 --dataset_path "./data/ImageNet-100" --dataset_name "imagenet100"
python precompute_embeddings.py --model_id "ViT-L/14@336px" --batch_size 512 --dataset_path "./data/LFW" --dataset_name "lfw"

## ResNet-50
python precompute_embeddings.py --model_id "RN50" --batch_size 512 --dataset_path "./data/COCO" --dataset_name "coco"
python precompute_embeddings.py --model_id "RN50" --batch_size 512 --dataset_path "./data/FACET" --dataset_name "facet"
python precompute_embeddings.py --model_id "RN50" --batch_size 512 --dataset_path "./data/ImageNet-100" --dataset_name "imagenet100"
python precompute_embeddings.py --model_id "RN50" --batch_size 512 --dataset_path "./data/LFW" --dataset_name "lfw"

## ResNet-101
python precompute_embeddings.py --model_id "RN101" --batch_size 512 --dataset_path "./data/COCO" --dataset_name "coco"
python precompute_embeddings.py --model_id "RN101" --batch_size 512 --dataset_path "./data/FACET" --dataset_name "facet"
python precompute_embeddings.py --model_id "RN101" --batch_size 512 --dataset_path "./data/ImageNet-100" --dataset_name "imagenet100"
python precompute_embeddings.py --model_id "RN101" --batch_size 512 --dataset_path "./data/LFW" --dataset_name "lfw"
