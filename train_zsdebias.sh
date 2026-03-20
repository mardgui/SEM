# #!/bin/bash

# 1. Train Demographic Bias Model (Gender/Race) (ViT-B/16)
# Uses: LFW, UTKFace, FACET
# Bias Attribute: 'human'
echo "------------------------------------------------"
echo "Starting Demographic Bias Training for ViT-B/16 (seed 0)..."
echo "------------------------------------------------"
python debiasing_methods/zsdebias_training.py \
  --dataset_paths data/LFW data/UTKFace data/FACET \
  --bias_attribute human \
  --clip_model ViT-B/16 \
  --epochs 500 \
  --patience 50 \
  --bs 1024 \
  --lr 1e-4 \
  --weight_decay 1e-5 \
  --seed 0 \
  --use_wandb
echo "------------------------------------------------"
echo "Starting Demographic Bias Training for ViT-B/16 (seed 1)..."
echo "------------------------------------------------"
python debiasing_methods/zsdebias_training.py \
  --dataset_paths data/LFW data/UTKFace data/FACET \
  --bias_attribute human \
  --clip_model ViT-B/16 \
  --epochs 500 \
  --patience 50 \
  --bs 1024 \
  --lr 1e-4 \
  --weight_decay 1e-5 \
  --seed 1 \
  --use_wandb
echo "------------------------------------------------"
echo "Starting Demographic Bias Training for ViT-B/16 (seed 2)..."
echo "------------------------------------------------"
python debiasing_methods/zsdebias_training.py \
  --dataset_paths data/LFW data/UTKFace data/FACET \
  --bias_attribute human \
  --clip_model ViT-B/16 \
  --epochs 500 \
  --patience 50 \
  --bs 1024 \
  --lr 1e-4 \
  --weight_decay 1e-5 \
  --seed 2 \
  --use_wandb

# 2. Train Background Bias Model (ViT-B/16)
# Uses: ImageNet-100, COCO
# Bias Attribute: 'animal'
echo "------------------------------------------------"
echo "Starting Background Bias Training for ViT-B/16 (seed 0)..."
echo "------------------------------------------------"
python debiasing_methods/zsdebias_training.py \
  --dataset_paths data/ImageNet-100 data/COCO \
  --bias_attribute animal \
  --clip_model ViT-B/16 \
  --epochs 500 \
  --patience 50 \
  --bs 1024 \
  --lr 1e-4 \
  --weight_decay 1e-5 \
  --seed 0 \
  --use_wandb
echo "------------------------------------------------"
echo "Starting Background Bias Training for ViT-B/16 (seed 1)..."
echo "------------------------------------------------"
python debiasing_methods/zsdebias_training.py \
  --dataset_paths data/ImageNet-100 data/COCO \
  --bias_attribute animal \
  --clip_model ViT-B/16 \
  --epochs 500 \
  --patience 50 \
  --bs 1024 \
  --lr 1e-4 \
  --weight_decay 1e-5 \
  --seed 1 \
  --use_wandb
echo "------------------------------------------------"
echo "Starting Background Bias Training for ViT-B/16 (seed 2)..."
echo "------------------------------------------------"
python debiasing_methods/zsdebias_training.py \
  --dataset_paths data/ImageNet-100 data/COCO \
  --bias_attribute animal \
  --clip_model ViT-B/16 \
  --epochs 500 \
  --patience 50 \
  --bs 1024 \
  --lr 1e-4 \
  --weight_decay 1e-5 \
  --seed 2 \
  --use_wandb

# 3. Train Demographic Bias Model (Gender/Race) (ViT-L/14@336px)
# Uses: LFW, UTKFace, FACET
# Bias Attribute: 'human'
echo "------------------------------------------------"
echo "Starting Demographic Bias Training for ViT-L/14@336px (seed 0)..."
echo "------------------------------------------------"
python debiasing_methods/zsdebias_training.py \
  --dataset_paths data/LFW data/UTKFace data/FACET \
  --bias_attribute human \
  --clip_model ViT-L/14@336px \
  --epochs 500 \
  --patience 50 \
  --bs 1024 \
  --lr 1e-4 \
  --weight_decay 1e-5 \
  --seed 0 \
  --use_wandb
echo "------------------------------------------------"
echo "Starting Demographic Bias Training for ViT-L/14@336px (seed 1)..."
echo "------------------------------------------------"
python debiasing_methods/zsdebias_training.py \
  --dataset_paths data/LFW data/UTKFace data/FACET \
  --bias_attribute human \
  --clip_model ViT-L/14@336px \
  --epochs 500 \
  --patience 50 \
  --bs 1024 \
  --lr 1e-4 \
  --weight_decay 1e-5 \
  --seed 1 \
  --use_wandb
echo "------------------------------------------------"
echo "Starting Demographic Bias Training for ViT-L/14@336px (seed 2)..."
echo "------------------------------------------------"
python debiasing_methods/zsdebias_training.py \
  --dataset_paths data/LFW data/UTKFace data/FACET \
  --bias_attribute human \
  --clip_model ViT-L/14@336px \
  --epochs 500 \
  --patience 50 \
  --bs 1024 \
  --lr 1e-4 \
  --weight_decay 1e-5 \
  --seed 2 \
  --use_wandb

# 4. Train Background Bias Model (ViT-L/14@336px)
# Uses: ImageNet-100, COCO
# Bias Attribute: 'animal'
echo "------------------------------------------------"
echo "Starting Background Bias Training for ViT-L/14@336px (seed 0)..."
echo "------------------------------------------------"
python debiasing_methods/zsdebias_training.py \
  --dataset_paths data/ImageNet-100 data/COCO \
  --bias_attribute animal \
  --clip_model ViT-L/14@336px \
  --epochs 500 \
  --patience 50 \
  --bs 1024 \
  --lr 1e-4 \
  --weight_decay 1e-5 \
  --seed 0 \
  --use_wandb
echo "------------------------------------------------"
echo "Starting Background Bias Training for ViT-L/14@336px (seed 1)..."
echo "------------------------------------------------"
python debiasing_methods/zsdebias_training.py \
  --dataset_paths data/ImageNet-100 data/COCO \
  --bias_attribute animal \
  --clip_model ViT-L/14@336px \
  --epochs 500 \
  --patience 50 \
  --bs 1024 \
  --lr 1e-4 \
  --weight_decay 1e-5 \
  --seed 1 \
  --use_wandb
echo "------------------------------------------------"
echo "Starting Background Bias Training for ViT-L/14@336px (seed 2)..."
echo "------------------------------------------------"
python debiasing_methods/zsdebias_training.py \
  --dataset_paths data/ImageNet-100 data/COCO \
  --bias_attribute animal \
  --clip_model ViT-L/14@336px \
  --epochs 500 \
  --patience 50 \
  --bs 1024 \
  --lr 1e-4 \
  --weight_decay 1e-5 \
  --seed 2 \
  --use_wandb

# 5. Train Demographic Bias Model (Gender/Race) (RN50)
# Uses: LFW, UTKFace, FACET
# Bias Attribute: 'human'
echo "------------------------------------------------"
echo "Starting Demographic Bias Training for ResNet-50 (seed 0)..."
echo "------------------------------------------------"
python debiasing_methods/zsdebias_training.py \
  --dataset_paths data/LFW data/UTKFace data/FACET \
  --bias_attribute human \
  --clip_model RN50 \
  --epochs 500 \
  --patience 50 \
  --bs 1024 \
  --lr 1e-4 \
  --weight_decay 1e-5 \
  --seed 0 \
  --use_wandb
echo "------------------------------------------------"
echo "Starting Demographic Bias Training for ResNet-50 (seed 1)..."
echo "------------------------------------------------"
python debiasing_methods/zsdebias_training.py \
  --dataset_paths data/LFW data/UTKFace data/FACET \
  --bias_attribute human \
  --clip_model RN50 \
  --epochs 500 \
  --patience 50 \
  --bs 1024 \
  --lr 1e-4 \
  --weight_decay 1e-5 \
  --seed 1 \
  --use_wandb
echo "------------------------------------------------"
echo "Starting Demographic Bias Training for ResNet-50 (seed 2)..."
echo "------------------------------------------------"
python debiasing_methods/zsdebias_training.py \
  --dataset_paths data/LFW data/UTKFace data/FACET \
  --bias_attribute human \
  --clip_model RN50 \
  --epochs 500 \
  --patience 50 \
  --bs 1024 \
  --lr 1e-4 \
  --weight_decay 1e-5 \
  --seed 2 \
  --use_wandb

# 6. Train Background Bias Model (RN50)
# Uses: ImageNet-100, COCO
# Bias Attribute: 'animal'
echo "------------------------------------------------"
echo "Starting Background Bias Training for ResNet-50 (seed 0)..."
echo "------------------------------------------------"
python debiasing_methods/zsdebias_training.py \
  --dataset_paths data/ImageNet-100 data/COCO \
  --bias_attribute animal \
  --clip_model RN50 \
  --epochs 500 \
  --patience 50 \
  --bs 1024 \
  --lr 1e-4 \
  --weight_decay 1e-5 \
  --seed 0 \
  --use_wandb
echo "------------------------------------------------"
echo "Starting Background Bias Training for ResNet-50 (seed 1)..."
echo "------------------------------------------------"
python debiasing_methods/zsdebias_training.py \
  --dataset_paths data/ImageNet-100 data/COCO \
  --bias_attribute animal \
  --clip_model RN50 \
  --epochs 500 \
  --patience 50 \
  --bs 1024 \
  --lr 1e-4 \
  --weight_decay 1e-5 \
  --seed 1 \
  --use_wandb
echo "------------------------------------------------"
echo "Starting Background Bias Training for ResNet-50 (seed 2)..."
echo "------------------------------------------------"
python debiasing_methods/zsdebias_training.py \
  --dataset_paths data/ImageNet-100 data/COCO \
  --bias_attribute animal \
  --clip_model RN50 \
  --epochs 500 \
  --patience 50 \
  --bs 1024 \
  --lr 1e-4 \
  --weight_decay 1e-5 \
  --seed 2 \
  --use_wandb

# 7. Train Demographic Bias Model (Gender/Race) (RN101)
# Uses: LFW, UTKFace, FACET
# Bias Attribute: 'human'
echo "------------------------------------------------"
echo "Starting Demographic Bias Training for ResNet-101 (seed 0)..."
echo "------------------------------------------------"
python debiasing_methods/zsdebias_training.py \
  --dataset_paths data/LFW data/UTKFace data/FACET \
  --bias_attribute human \
  --clip_model RN101 \
  --epochs 500 \
  --patience 50 \
  --bs 1024 \
  --lr 1e-4 \
  --weight_decay 1e-5 \
  --seed 0 \
  --use_wandb
echo "------------------------------------------------"
echo "Starting Demographic Bias Training for ResNet-101 (seed 1)..."
echo "------------------------------------------------"
python debiasing_methods/zsdebias_training.py \
  --dataset_paths data/LFW data/UTKFace data/FACET \
  --bias_attribute human \
  --clip_model RN101 \
  --epochs 500 \
  --patience 50 \
  --bs 1024 \
  --lr 1e-4 \
  --weight_decay 1e-5 \
  --seed 1 \
  --use_wandb
echo "------------------------------------------------"
echo "Starting Demographic Bias Training for ResNet-101 (seed 2)..."
echo "------------------------------------------------"
python debiasing_methods/zsdebias_training.py \
  --dataset_paths data/LFW data/UTKFace data/FACET \
  --bias_attribute human \
  --clip_model RN101 \
  --epochs 500 \
  --patience 50 \
  --bs 1024 \
  --lr 1e-4 \
  --weight_decay 1e-5 \
  --seed 2 \
  --use_wandb

# 8. Train Background Bias Model (RN101)
# Uses: ImageNet-100, COCO
# Bias Attribute: 'animal'
echo "------------------------------------------------"
echo "Starting Background Bias Training for ResNet-101 (seed 0)..."
echo "------------------------------------------------"
python debiasing_methods/zsdebias_training.py \
  --dataset_paths data/ImageNet-100 data/COCO \
  --bias_attribute animal \
  --clip_model RN101 \
  --epochs 500 \
  --patience 50 \
  --bs 1024 \
  --lr 1e-4 \
  --weight_decay 1e-5 \
  --seed 0 \
  --use_wandb
echo "------------------------------------------------"
echo "Starting Background Bias Training for ResNet-101 (seed 1)..."
echo "------------------------------------------------"
python debiasing_methods/zsdebias_training.py \
  --dataset_paths data/ImageNet-100 data/COCO \
  --bias_attribute animal \
  --clip_model RN101 \
  --epochs 500 \
  --patience 50 \
  --bs 1024 \
  --lr 1e-4 \
  --weight_decay 1e-5 \
  --seed 1 \
  --use_wandb
echo "------------------------------------------------"
echo "Starting Background Bias Training for ResNet-101 (seed 2)..."
echo "------------------------------------------------"
python debiasing_methods/zsdebias_training.py \
  --dataset_paths data/ImageNet-100 data/COCO \
  --bias_attribute animal \
  --clip_model RN101 \
  --epochs 500 \
  --patience 50 \
  --bs 1024 \
  --lr 1e-4 \
  --weight_decay 1e-5 \
  --seed 2 \
  --use_wandb
