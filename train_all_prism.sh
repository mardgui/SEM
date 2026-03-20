# Retrieval tasks
## Backbone: ViT-B/16
### Setting: stereotype_addict_gender
python debiasing_methods/prism_training.py --CLIP_model ViT-B/16 --task retrieval --setting stereotype_addict_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-B/16 --task retrieval --setting stereotype_addict_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-B/16 --task retrieval --setting stereotype_addict_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_addict_race
python debiasing_methods/prism_training.py --CLIP_model ViT-B/16 --task retrieval --setting stereotype_addict_race --epochs 1 --num_bases 0 --num_samples 200 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-B/16 --task retrieval --setting stereotype_addict_race --epochs 1 --num_bases 0 --num_samples 200 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-B/16 --task retrieval --setting stereotype_addict_race --epochs 1 --num_bases 0 --num_samples 200 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_addict_utkrace
python debiasing_methods/prism_training.py --CLIP_model ViT-B/16 --task retrieval --setting stereotype_addict_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-B/16 --task retrieval --setting stereotype_addict_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-B/16 --task retrieval --setting stereotype_addict_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_burglar_gender
python debiasing_methods/prism_training.py --CLIP_model ViT-B/16 --task retrieval --setting stereotype_burglar_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-B/16 --task retrieval --setting stereotype_burglar_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-B/16 --task retrieval --setting stereotype_burglar_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_burglar_race
python debiasing_methods/prism_training.py --CLIP_model ViT-B/16 --task retrieval --setting stereotype_burglar_race --epochs 1 --num_bases 0 --num_samples 200 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-B/16 --task retrieval --setting stereotype_burglar_race --epochs 1 --num_bases 0 --num_samples 200 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-B/16 --task retrieval --setting stereotype_burglar_race --epochs 1 --num_bases 0 --num_samples 200 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_burglar_utkrace
python debiasing_methods/prism_training.py --CLIP_model ViT-B/16 --task retrieval --setting stereotype_burglar_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-B/16 --task retrieval --setting stereotype_burglar_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-B/16 --task retrieval --setting stereotype_burglar_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_criminal_gender
python debiasing_methods/prism_training.py --CLIP_model ViT-B/16 --task retrieval --setting stereotype_criminal_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-B/16 --task retrieval --setting stereotype_criminal_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-B/16 --task retrieval --setting stereotype_criminal_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_criminal_race
python debiasing_methods/prism_training.py --CLIP_model ViT-B/16 --task retrieval --setting stereotype_criminal_race --epochs 1 --num_bases 0 --num_samples 200 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-B/16 --task retrieval --setting stereotype_criminal_race --epochs 1 --num_bases 0 --num_samples 200 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-B/16 --task retrieval --setting stereotype_criminal_race --epochs 1 --num_bases 0 --num_samples 200 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_criminal_utkrace
python debiasing_methods/prism_training.py --CLIP_model ViT-B/16 --task retrieval --setting stereotype_criminal_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-B/16 --task retrieval --setting stereotype_criminal_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-B/16 --task retrieval --setting stereotype_criminal_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_dangerous-person_gender
python debiasing_methods/prism_training.py --CLIP_model ViT-B/16 --task retrieval --setting stereotype_dangerous-person_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-B/16 --task retrieval --setting stereotype_dangerous-person_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-B/16 --task retrieval --setting stereotype_dangerous-person_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_dangerous-person_race
python debiasing_methods/prism_training.py --CLIP_model ViT-B/16 --task retrieval --setting stereotype_dangerous-person_race --epochs 1 --num_bases 0 --num_samples 200 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-B/16 --task retrieval --setting stereotype_dangerous-person_race --epochs 1 --num_bases 0 --num_samples 200 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-B/16 --task retrieval --setting stereotype_dangerous-person_race --epochs 1 --num_bases 0 --num_samples 200 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_dangerous-person_utkrace
python debiasing_methods/prism_training.py --CLIP_model ViT-B/16 --task retrieval --setting stereotype_dangerous-person_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-B/16 --task retrieval --setting stereotype_dangerous-person_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-B/16 --task retrieval --setting stereotype_dangerous-person_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_delinquent_gender
python debiasing_methods/prism_training.py --CLIP_model ViT-B/16 --task retrieval --setting stereotype_delinquent_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-B/16 --task retrieval --setting stereotype_delinquent_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-B/16 --task retrieval --setting stereotype_delinquent_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_delinquent_race
python debiasing_methods/prism_training.py --CLIP_model ViT-B/16 --task retrieval --setting stereotype_delinquent_race --epochs 1 --num_bases 0 --num_samples 200 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-B/16 --task retrieval --setting stereotype_delinquent_race --epochs 1 --num_bases 0 --num_samples 200 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-B/16 --task retrieval --setting stereotype_delinquent_race --epochs 1 --num_bases 0 --num_samples 200 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_delinquent_utkrace
python debiasing_methods/prism_training.py --CLIP_model ViT-B/16 --task retrieval --setting stereotype_delinquent_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-B/16 --task retrieval --setting stereotype_delinquent_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-B/16 --task retrieval --setting stereotype_delinquent_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_deviant_gender
python debiasing_methods/prism_training.py --CLIP_model ViT-B/16 --task retrieval --setting stereotype_deviant_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-B/16 --task retrieval --setting stereotype_deviant_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-B/16 --task retrieval --setting stereotype_deviant_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_deviant_race
python debiasing_methods/prism_training.py --CLIP_model ViT-B/16 --task retrieval --setting stereotype_deviant_race --epochs 1 --num_bases 0 --num_samples 200 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-B/16 --task retrieval --setting stereotype_deviant_race --epochs 1 --num_bases 0 --num_samples 200 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-B/16 --task retrieval --setting stereotype_deviant_race --epochs 1 --num_bases 0 --num_samples 200 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_deviant_utkrace
python debiasing_methods/prism_training.py --CLIP_model ViT-B/16 --task retrieval --setting stereotype_deviant_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-B/16 --task retrieval --setting stereotype_deviant_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-B/16 --task retrieval --setting stereotype_deviant_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_embezzler_gender
python debiasing_methods/prism_training.py --CLIP_model ViT-B/16 --task retrieval --setting stereotype_embezzler_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-B/16 --task retrieval --setting stereotype_embezzler_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-B/16 --task retrieval --setting stereotype_embezzler_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_embezzler_race
python debiasing_methods/prism_training.py --CLIP_model ViT-B/16 --task retrieval --setting stereotype_embezzler_race --epochs 1 --num_bases 0 --num_samples 200 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-B/16 --task retrieval --setting stereotype_embezzler_race --epochs 1 --num_bases 0 --num_samples 200 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-B/16 --task retrieval --setting stereotype_embezzler_race --epochs 1 --num_bases 0 --num_samples 200 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_embezzler_utkrace
python debiasing_methods/prism_training.py --CLIP_model ViT-B/16 --task retrieval --setting stereotype_embezzler_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-B/16 --task retrieval --setting stereotype_embezzler_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-B/16 --task retrieval --setting stereotype_embezzler_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_felon_gender
python debiasing_methods/prism_training.py --CLIP_model ViT-B/16 --task retrieval --setting stereotype_felon_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-B/16 --task retrieval --setting stereotype_felon_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-B/16 --task retrieval --setting stereotype_felon_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_felon_race
python debiasing_methods/prism_training.py --CLIP_model ViT-B/16 --task retrieval --setting stereotype_felon_race --epochs 1 --num_bases 0 --num_samples 200 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-B/16 --task retrieval --setting stereotype_felon_race --epochs 1 --num_bases 0 --num_samples 200 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-B/16 --task retrieval --setting stereotype_felon_race --epochs 1 --num_bases 0 --num_samples 200 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_felon_utkrace
python debiasing_methods/prism_training.py --CLIP_model ViT-B/16 --task retrieval --setting stereotype_felon_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-B/16 --task retrieval --setting stereotype_felon_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-B/16 --task retrieval --setting stereotype_felon_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_fraud_gender
python debiasing_methods/prism_training.py --CLIP_model ViT-B/16 --task retrieval --setting stereotype_fraud_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-B/16 --task retrieval --setting stereotype_fraud_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-B/16 --task retrieval --setting stereotype_fraud_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_fraud_race
python debiasing_methods/prism_training.py --CLIP_model ViT-B/16 --task retrieval --setting stereotype_fraud_race --epochs 1 --num_bases 0 --num_samples 200 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-B/16 --task retrieval --setting stereotype_fraud_race --epochs 1 --num_bases 0 --num_samples 200 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-B/16 --task retrieval --setting stereotype_fraud_race --epochs 1 --num_bases 0 --num_samples 200 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_fraud_utkrace
python debiasing_methods/prism_training.py --CLIP_model ViT-B/16 --task retrieval --setting stereotype_fraud_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-B/16 --task retrieval --setting stereotype_fraud_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-B/16 --task retrieval --setting stereotype_fraud_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_freak_gender
python debiasing_methods/prism_training.py --CLIP_model ViT-B/16 --task retrieval --setting stereotype_freak_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-B/16 --task retrieval --setting stereotype_freak_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-B/16 --task retrieval --setting stereotype_freak_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_freak_race
python debiasing_methods/prism_training.py --CLIP_model ViT-B/16 --task retrieval --setting stereotype_freak_race --epochs 1 --num_bases 0 --num_samples 200 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-B/16 --task retrieval --setting stereotype_freak_race --epochs 1 --num_bases 0 --num_samples 200 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-B/16 --task retrieval --setting stereotype_freak_race --epochs 1 --num_bases 0 --num_samples 200 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_freak_utkrace
python debiasing_methods/prism_training.py --CLIP_model ViT-B/16 --task retrieval --setting stereotype_freak_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-B/16 --task retrieval --setting stereotype_freak_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-B/16 --task retrieval --setting stereotype_freak_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_gangster_gender
python debiasing_methods/prism_training.py --CLIP_model ViT-B/16 --task retrieval --setting stereotype_gangster_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-B/16 --task retrieval --setting stereotype_gangster_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-B/16 --task retrieval --setting stereotype_gangster_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_gangster_race
python debiasing_methods/prism_training.py --CLIP_model ViT-B/16 --task retrieval --setting stereotype_gangster_race --epochs 1 --num_bases 0 --num_samples 200 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-B/16 --task retrieval --setting stereotype_gangster_race --epochs 1 --num_bases 0 --num_samples 200 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-B/16 --task retrieval --setting stereotype_gangster_race --epochs 1 --num_bases 0 --num_samples 200 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_gangster_utkrace
python debiasing_methods/prism_training.py --CLIP_model ViT-B/16 --task retrieval --setting stereotype_gangster_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-B/16 --task retrieval --setting stereotype_gangster_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-B/16 --task retrieval --setting stereotype_gangster_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_hacker_gender
python debiasing_methods/prism_training.py --CLIP_model ViT-B/16 --task retrieval --setting stereotype_hacker_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-B/16 --task retrieval --setting stereotype_hacker_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-B/16 --task retrieval --setting stereotype_hacker_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_hacker_race
python debiasing_methods/prism_training.py --CLIP_model ViT-B/16 --task retrieval --setting stereotype_hacker_race --epochs 1 --num_bases 0 --num_samples 200 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-B/16 --task retrieval --setting stereotype_hacker_race --epochs 1 --num_bases 0 --num_samples 200 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-B/16 --task retrieval --setting stereotype_hacker_race --epochs 1 --num_bases 0 --num_samples 200 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_hacker_utkrace
python debiasing_methods/prism_training.py --CLIP_model ViT-B/16 --task retrieval --setting stereotype_hacker_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-B/16 --task retrieval --setting stereotype_hacker_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-B/16 --task retrieval --setting stereotype_hacker_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_illegal-person_gender
python debiasing_methods/prism_training.py --CLIP_model ViT-B/16 --task retrieval --setting stereotype_illegal-person_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-B/16 --task retrieval --setting stereotype_illegal-person_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-B/16 --task retrieval --setting stereotype_illegal-person_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_illegal-person_race
python debiasing_methods/prism_training.py --CLIP_model ViT-B/16 --task retrieval --setting stereotype_illegal-person_race --epochs 1 --num_bases 0 --num_samples 200 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-B/16 --task retrieval --setting stereotype_illegal-person_race --epochs 1 --num_bases 0 --num_samples 200 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-B/16 --task retrieval --setting stereotype_illegal-person_race --epochs 1 --num_bases 0 --num_samples 200 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_illegal-person_utkrace
python debiasing_methods/prism_training.py --CLIP_model ViT-B/16 --task retrieval --setting stereotype_illegal-person_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-B/16 --task retrieval --setting stereotype_illegal-person_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-B/16 --task retrieval --setting stereotype_illegal-person_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_insane-person_gender
python debiasing_methods/prism_training.py --CLIP_model ViT-B/16 --task retrieval --setting stereotype_insane-person_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-B/16 --task retrieval --setting stereotype_insane-person_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-B/16 --task retrieval --setting stereotype_insane-person_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_insane-person_race
python debiasing_methods/prism_training.py --CLIP_model ViT-B/16 --task retrieval --setting stereotype_insane-person_race --epochs 1 --num_bases 0 --num_samples 200 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-B/16 --task retrieval --setting stereotype_insane-person_race --epochs 1 --num_bases 0 --num_samples 200 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-B/16 --task retrieval --setting stereotype_insane-person_race --epochs 1 --num_bases 0 --num_samples 200 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_insane-person_utkrace
python debiasing_methods/prism_training.py --CLIP_model ViT-B/16 --task retrieval --setting stereotype_insane-person_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-B/16 --task retrieval --setting stereotype_insane-person_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-B/16 --task retrieval --setting stereotype_insane-person_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_lawless-person_gender
python debiasing_methods/prism_training.py --CLIP_model ViT-B/16 --task retrieval --setting stereotype_lawless-person_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-B/16 --task retrieval --setting stereotype_lawless-person_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-B/16 --task retrieval --setting stereotype_lawless-person_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_lawless-person_race
python debiasing_methods/prism_training.py --CLIP_model ViT-B/16 --task retrieval --setting stereotype_lawless-person_race --epochs 1 --num_bases 0 --num_samples 200 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-B/16 --task retrieval --setting stereotype_lawless-person_race --epochs 1 --num_bases 0 --num_samples 200 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-B/16 --task retrieval --setting stereotype_lawless-person_race --epochs 1 --num_bases 0 --num_samples 200 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_lawless-person_utkrace
python debiasing_methods/prism_training.py --CLIP_model ViT-B/16 --task retrieval --setting stereotype_lawless-person_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-B/16 --task retrieval --setting stereotype_lawless-person_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-B/16 --task retrieval --setting stereotype_lawless-person_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_mugger_gender
python debiasing_methods/prism_training.py --CLIP_model ViT-B/16 --task retrieval --setting stereotype_mugger_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-B/16 --task retrieval --setting stereotype_mugger_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-B/16 --task retrieval --setting stereotype_mugger_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_mugger_race
python debiasing_methods/prism_training.py --CLIP_model ViT-B/16 --task retrieval --setting stereotype_mugger_race --epochs 1 --num_bases 0 --num_samples 200 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-B/16 --task retrieval --setting stereotype_mugger_race --epochs 1 --num_bases 0 --num_samples 200 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-B/16 --task retrieval --setting stereotype_mugger_race --epochs 1 --num_bases 0 --num_samples 200 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_mugger_utkrace
python debiasing_methods/prism_training.py --CLIP_model ViT-B/16 --task retrieval --setting stereotype_mugger_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-B/16 --task retrieval --setting stereotype_mugger_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-B/16 --task retrieval --setting stereotype_mugger_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_murderer_gender
python debiasing_methods/prism_training.py --CLIP_model ViT-B/16 --task retrieval --setting stereotype_murderer_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-B/16 --task retrieval --setting stereotype_murderer_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-B/16 --task retrieval --setting stereotype_murderer_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_murderer_race
python debiasing_methods/prism_training.py --CLIP_model ViT-B/16 --task retrieval --setting stereotype_murderer_race --epochs 1 --num_bases 0 --num_samples 200 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-B/16 --task retrieval --setting stereotype_murderer_race --epochs 1 --num_bases 0 --num_samples 200 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-B/16 --task retrieval --setting stereotype_murderer_race --epochs 1 --num_bases 0 --num_samples 200 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_murderer_utkrace
python debiasing_methods/prism_training.py --CLIP_model ViT-B/16 --task retrieval --setting stereotype_murderer_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-B/16 --task retrieval --setting stereotype_murderer_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-B/16 --task retrieval --setting stereotype_murderer_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_psychopath_gender
python debiasing_methods/prism_training.py --CLIP_model ViT-B/16 --task retrieval --setting stereotype_psychopath_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-B/16 --task retrieval --setting stereotype_psychopath_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-B/16 --task retrieval --setting stereotype_psychopath_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_psychopath_race
python debiasing_methods/prism_training.py --CLIP_model ViT-B/16 --task retrieval --setting stereotype_psychopath_race --epochs 1 --num_bases 0 --num_samples 200 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-B/16 --task retrieval --setting stereotype_psychopath_race --epochs 1 --num_bases 0 --num_samples 200 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-B/16 --task retrieval --setting stereotype_psychopath_race --epochs 1 --num_bases 0 --num_samples 200 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_psychopath_utkrace
python debiasing_methods/prism_training.py --CLIP_model ViT-B/16 --task retrieval --setting stereotype_psychopath_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-B/16 --task retrieval --setting stereotype_psychopath_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-B/16 --task retrieval --setting stereotype_psychopath_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_robber_gender
python debiasing_methods/prism_training.py --CLIP_model ViT-B/16 --task retrieval --setting stereotype_robber_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-B/16 --task retrieval --setting stereotype_robber_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-B/16 --task retrieval --setting stereotype_robber_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_robber_race
python debiasing_methods/prism_training.py --CLIP_model ViT-B/16 --task retrieval --setting stereotype_robber_race --epochs 1 --num_bases 0 --num_samples 200 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-B/16 --task retrieval --setting stereotype_robber_race --epochs 1 --num_bases 0 --num_samples 200 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-B/16 --task retrieval --setting stereotype_robber_race --epochs 1 --num_bases 0 --num_samples 200 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_robber_utkrace
python debiasing_methods/prism_training.py --CLIP_model ViT-B/16 --task retrieval --setting stereotype_robber_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-B/16 --task retrieval --setting stereotype_robber_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-B/16 --task retrieval --setting stereotype_robber_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_savage_gender
python debiasing_methods/prism_training.py --CLIP_model ViT-B/16 --task retrieval --setting stereotype_savage_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-B/16 --task retrieval --setting stereotype_savage_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-B/16 --task retrieval --setting stereotype_savage_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_savage_race
python debiasing_methods/prism_training.py --CLIP_model ViT-B/16 --task retrieval --setting stereotype_savage_race --epochs 1 --num_bases 0 --num_samples 200 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-B/16 --task retrieval --setting stereotype_savage_race --epochs 1 --num_bases 0 --num_samples 200 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-B/16 --task retrieval --setting stereotype_savage_race --epochs 1 --num_bases 0 --num_samples 200 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_savage_utkrace
python debiasing_methods/prism_training.py --CLIP_model ViT-B/16 --task retrieval --setting stereotype_savage_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-B/16 --task retrieval --setting stereotype_savage_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-B/16 --task retrieval --setting stereotype_savage_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_shoplifter_gender
python debiasing_methods/prism_training.py --CLIP_model ViT-B/16 --task retrieval --setting stereotype_shoplifter_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-B/16 --task retrieval --setting stereotype_shoplifter_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-B/16 --task retrieval --setting stereotype_shoplifter_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_shoplifter_race
python debiasing_methods/prism_training.py --CLIP_model ViT-B/16 --task retrieval --setting stereotype_shoplifter_race --epochs 1 --num_bases 0 --num_samples 200 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-B/16 --task retrieval --setting stereotype_shoplifter_race --epochs 1 --num_bases 0 --num_samples 200 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-B/16 --task retrieval --setting stereotype_shoplifter_race --epochs 1 --num_bases 0 --num_samples 200 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_shoplifter_utkrace
python debiasing_methods/prism_training.py --CLIP_model ViT-B/16 --task retrieval --setting stereotype_shoplifter_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-B/16 --task retrieval --setting stereotype_shoplifter_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-B/16 --task retrieval --setting stereotype_shoplifter_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_terrorist_gender
python debiasing_methods/prism_training.py --CLIP_model ViT-B/16 --task retrieval --setting stereotype_terrorist_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-B/16 --task retrieval --setting stereotype_terrorist_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-B/16 --task retrieval --setting stereotype_terrorist_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_terrorist_race
python debiasing_methods/prism_training.py --CLIP_model ViT-B/16 --task retrieval --setting stereotype_terrorist_race --epochs 1 --num_bases 0 --num_samples 200 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-B/16 --task retrieval --setting stereotype_terrorist_race --epochs 1 --num_bases 0 --num_samples 200 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-B/16 --task retrieval --setting stereotype_terrorist_race --epochs 1 --num_bases 0 --num_samples 200 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_terrorist_utkrace
python debiasing_methods/prism_training.py --CLIP_model ViT-B/16 --task retrieval --setting stereotype_terrorist_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-B/16 --task retrieval --setting stereotype_terrorist_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-B/16 --task retrieval --setting stereotype_terrorist_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_thief_gender
python debiasing_methods/prism_training.py --CLIP_model ViT-B/16 --task retrieval --setting stereotype_thief_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-B/16 --task retrieval --setting stereotype_thief_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-B/16 --task retrieval --setting stereotype_thief_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_thief_race
python debiasing_methods/prism_training.py --CLIP_model ViT-B/16 --task retrieval --setting stereotype_thief_race --epochs 1 --num_bases 0 --num_samples 200 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-B/16 --task retrieval --setting stereotype_thief_race --epochs 1 --num_bases 0 --num_samples 200 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-B/16 --task retrieval --setting stereotype_thief_race --epochs 1 --num_bases 0 --num_samples 200 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_thief_utkrace
python debiasing_methods/prism_training.py --CLIP_model ViT-B/16 --task retrieval --setting stereotype_thief_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-B/16 --task retrieval --setting stereotype_thief_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-B/16 --task retrieval --setting stereotype_thief_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_thug_gender
python debiasing_methods/prism_training.py --CLIP_model ViT-B/16 --task retrieval --setting stereotype_thug_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-B/16 --task retrieval --setting stereotype_thug_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-B/16 --task retrieval --setting stereotype_thug_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_thug_race
python debiasing_methods/prism_training.py --CLIP_model ViT-B/16 --task retrieval --setting stereotype_thug_race --epochs 1 --num_bases 0 --num_samples 200 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-B/16 --task retrieval --setting stereotype_thug_race --epochs 1 --num_bases 0 --num_samples 200 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-B/16 --task retrieval --setting stereotype_thug_race --epochs 1 --num_bases 0 --num_samples 200 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_thug_utkrace
python debiasing_methods/prism_training.py --CLIP_model ViT-B/16 --task retrieval --setting stereotype_thug_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-B/16 --task retrieval --setting stereotype_thug_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-B/16 --task retrieval --setting stereotype_thug_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_violent-person_gender
python debiasing_methods/prism_training.py --CLIP_model ViT-B/16 --task retrieval --setting stereotype_violent-person_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-B/16 --task retrieval --setting stereotype_violent-person_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-B/16 --task retrieval --setting stereotype_violent-person_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_violent-person_race
python debiasing_methods/prism_training.py --CLIP_model ViT-B/16 --task retrieval --setting stereotype_violent-person_race --epochs 1 --num_bases 0 --num_samples 200 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-B/16 --task retrieval --setting stereotype_violent-person_race --epochs 1 --num_bases 0 --num_samples 200 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-B/16 --task retrieval --setting stereotype_violent-person_race --epochs 1 --num_bases 0 --num_samples 200 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_violent-person_utkrace
python debiasing_methods/prism_training.py --CLIP_model ViT-B/16 --task retrieval --setting stereotype_violent-person_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-B/16 --task retrieval --setting stereotype_violent-person_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-B/16 --task retrieval --setting stereotype_violent-person_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: celeba_black_hair_gender
python debiasing_methods/prism_training.py --CLIP_model ViT-B/16 --task retrieval --setting celeba_black_hair_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-B/16 --task retrieval --setting celeba_black_hair_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-B/16 --task retrieval --setting celeba_black_hair_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: celeba_blond_hair_gender
python debiasing_methods/prism_training.py --CLIP_model ViT-B/16 --task retrieval --setting celeba_blond_hair_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-B/16 --task retrieval --setting celeba_blond_hair_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-B/16 --task retrieval --setting celeba_blond_hair_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: celeba_brown_hair_gender
python debiasing_methods/prism_training.py --CLIP_model ViT-B/16 --task retrieval --setting celeba_brown_hair_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-B/16 --task retrieval --setting celeba_brown_hair_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-B/16 --task retrieval --setting celeba_brown_hair_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: celeba_gray_hair_gender
python debiasing_methods/prism_training.py --CLIP_model ViT-B/16 --task retrieval --setting celeba_gray_hair_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-B/16 --task retrieval --setting celeba_gray_hair_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-B/16 --task retrieval --setting celeba_gray_hair_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random


## Backbone: ViT-L/14@336px
### Setting: stereotype_addict_gender
python debiasing_methods/prism_training.py --CLIP_model ViT-L/14@336px --task retrieval --setting stereotype_addict_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-L/14@336px --task retrieval --setting stereotype_addict_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-L/14@336px --task retrieval --setting stereotype_addict_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_addict_race
python debiasing_methods/prism_training.py --CLIP_model ViT-L/14@336px --task retrieval --setting stereotype_addict_race --epochs 1 --num_bases 0 --num_samples 200 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-L/14@336px --task retrieval --setting stereotype_addict_race --epochs 1 --num_bases 0 --num_samples 200 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-L/14@336px --task retrieval --setting stereotype_addict_race --epochs 1 --num_bases 0 --num_samples 200 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_addict_utkrace
python debiasing_methods/prism_training.py --CLIP_model ViT-L/14@336px --task retrieval --setting stereotype_addict_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-L/14@336px --task retrieval --setting stereotype_addict_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-L/14@336px --task retrieval --setting stereotype_addict_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_burglar_gender
python debiasing_methods/prism_training.py --CLIP_model ViT-L/14@336px --task retrieval --setting stereotype_burglar_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-L/14@336px --task retrieval --setting stereotype_burglar_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-L/14@336px --task retrieval --setting stereotype_burglar_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_burglar_race
python debiasing_methods/prism_training.py --CLIP_model ViT-L/14@336px --task retrieval --setting stereotype_burglar_race --epochs 1 --num_bases 0 --num_samples 200 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-L/14@336px --task retrieval --setting stereotype_burglar_race --epochs 1 --num_bases 0 --num_samples 200 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-L/14@336px --task retrieval --setting stereotype_burglar_race --epochs 1 --num_bases 0 --num_samples 200 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_burglar_utkrace
python debiasing_methods/prism_training.py --CLIP_model ViT-L/14@336px --task retrieval --setting stereotype_burglar_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-L/14@336px --task retrieval --setting stereotype_burglar_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-L/14@336px --task retrieval --setting stereotype_burglar_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_criminal_gender
python debiasing_methods/prism_training.py --CLIP_model ViT-L/14@336px --task retrieval --setting stereotype_criminal_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-L/14@336px --task retrieval --setting stereotype_criminal_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-L/14@336px --task retrieval --setting stereotype_criminal_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_criminal_race
python debiasing_methods/prism_training.py --CLIP_model ViT-L/14@336px --task retrieval --setting stereotype_criminal_race --epochs 1 --num_bases 0 --num_samples 200 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-L/14@336px --task retrieval --setting stereotype_criminal_race --epochs 1 --num_bases 0 --num_samples 200 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-L/14@336px --task retrieval --setting stereotype_criminal_race --epochs 1 --num_bases 0 --num_samples 200 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_criminal_utkrace
python debiasing_methods/prism_training.py --CLIP_model ViT-L/14@336px --task retrieval --setting stereotype_criminal_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-L/14@336px --task retrieval --setting stereotype_criminal_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-L/14@336px --task retrieval --setting stereotype_criminal_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_dangerous-person_gender
python debiasing_methods/prism_training.py --CLIP_model ViT-L/14@336px --task retrieval --setting stereotype_dangerous-person_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-L/14@336px --task retrieval --setting stereotype_dangerous-person_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-L/14@336px --task retrieval --setting stereotype_dangerous-person_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_dangerous-person_race
python debiasing_methods/prism_training.py --CLIP_model ViT-L/14@336px --task retrieval --setting stereotype_dangerous-person_race --epochs 1 --num_bases 0 --num_samples 200 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-L/14@336px --task retrieval --setting stereotype_dangerous-person_race --epochs 1 --num_bases 0 --num_samples 200 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-L/14@336px --task retrieval --setting stereotype_dangerous-person_race --epochs 1 --num_bases 0 --num_samples 200 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_dangerous-person_utkrace
python debiasing_methods/prism_training.py --CLIP_model ViT-L/14@336px --task retrieval --setting stereotype_dangerous-person_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-L/14@336px --task retrieval --setting stereotype_dangerous-person_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-L/14@336px --task retrieval --setting stereotype_dangerous-person_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_delinquent_gender
python debiasing_methods/prism_training.py --CLIP_model ViT-L/14@336px --task retrieval --setting stereotype_delinquent_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-L/14@336px --task retrieval --setting stereotype_delinquent_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-L/14@336px --task retrieval --setting stereotype_delinquent_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_delinquent_race
python debiasing_methods/prism_training.py --CLIP_model ViT-L/14@336px --task retrieval --setting stereotype_delinquent_race --epochs 1 --num_bases 0 --num_samples 200 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-L/14@336px --task retrieval --setting stereotype_delinquent_race --epochs 1 --num_bases 0 --num_samples 200 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-L/14@336px --task retrieval --setting stereotype_delinquent_race --epochs 1 --num_bases 0 --num_samples 200 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_delinquent_utkrace
python debiasing_methods/prism_training.py --CLIP_model ViT-L/14@336px --task retrieval --setting stereotype_delinquent_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-L/14@336px --task retrieval --setting stereotype_delinquent_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-L/14@336px --task retrieval --setting stereotype_delinquent_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_deviant_gender
python debiasing_methods/prism_training.py --CLIP_model ViT-L/14@336px --task retrieval --setting stereotype_deviant_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-L/14@336px --task retrieval --setting stereotype_deviant_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-L/14@336px --task retrieval --setting stereotype_deviant_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_deviant_race
python debiasing_methods/prism_training.py --CLIP_model ViT-L/14@336px --task retrieval --setting stereotype_deviant_race --epochs 1 --num_bases 0 --num_samples 200 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-L/14@336px --task retrieval --setting stereotype_deviant_race --epochs 1 --num_bases 0 --num_samples 200 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-L/14@336px --task retrieval --setting stereotype_deviant_race --epochs 1 --num_bases 0 --num_samples 200 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_deviant_utkrace
python debiasing_methods/prism_training.py --CLIP_model ViT-L/14@336px --task retrieval --setting stereotype_deviant_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-L/14@336px --task retrieval --setting stereotype_deviant_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-L/14@336px --task retrieval --setting stereotype_deviant_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_embezzler_gender
python debiasing_methods/prism_training.py --CLIP_model ViT-L/14@336px --task retrieval --setting stereotype_embezzler_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-L/14@336px --task retrieval --setting stereotype_embezzler_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-L/14@336px --task retrieval --setting stereotype_embezzler_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_embezzler_race
python debiasing_methods/prism_training.py --CLIP_model ViT-L/14@336px --task retrieval --setting stereotype_embezzler_race --epochs 1 --num_bases 0 --num_samples 200 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-L/14@336px --task retrieval --setting stereotype_embezzler_race --epochs 1 --num_bases 0 --num_samples 200 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-L/14@336px --task retrieval --setting stereotype_embezzler_race --epochs 1 --num_bases 0 --num_samples 200 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_embezzler_utkrace
python debiasing_methods/prism_training.py --CLIP_model ViT-L/14@336px --task retrieval --setting stereotype_embezzler_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-L/14@336px --task retrieval --setting stereotype_embezzler_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-L/14@336px --task retrieval --setting stereotype_embezzler_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_felon_gender
python debiasing_methods/prism_training.py --CLIP_model ViT-L/14@336px --task retrieval --setting stereotype_felon_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-L/14@336px --task retrieval --setting stereotype_felon_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-L/14@336px --task retrieval --setting stereotype_felon_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_felon_race
python debiasing_methods/prism_training.py --CLIP_model ViT-L/14@336px --task retrieval --setting stereotype_felon_race --epochs 1 --num_bases 0 --num_samples 200 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-L/14@336px --task retrieval --setting stereotype_felon_race --epochs 1 --num_bases 0 --num_samples 200 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-L/14@336px --task retrieval --setting stereotype_felon_race --epochs 1 --num_bases 0 --num_samples 200 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_felon_utkrace
python debiasing_methods/prism_training.py --CLIP_model ViT-L/14@336px --task retrieval --setting stereotype_felon_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-L/14@336px --task retrieval --setting stereotype_felon_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-L/14@336px --task retrieval --setting stereotype_felon_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_fraud_gender
python debiasing_methods/prism_training.py --CLIP_model ViT-L/14@336px --task retrieval --setting stereotype_fraud_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-L/14@336px --task retrieval --setting stereotype_fraud_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-L/14@336px --task retrieval --setting stereotype_fraud_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_fraud_race
python debiasing_methods/prism_training.py --CLIP_model ViT-L/14@336px --task retrieval --setting stereotype_fraud_race --epochs 1 --num_bases 0 --num_samples 200 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-L/14@336px --task retrieval --setting stereotype_fraud_race --epochs 1 --num_bases 0 --num_samples 200 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-L/14@336px --task retrieval --setting stereotype_fraud_race --epochs 1 --num_bases 0 --num_samples 200 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_fraud_utkrace
python debiasing_methods/prism_training.py --CLIP_model ViT-L/14@336px --task retrieval --setting stereotype_fraud_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-L/14@336px --task retrieval --setting stereotype_fraud_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-L/14@336px --task retrieval --setting stereotype_fraud_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_freak_gender
python debiasing_methods/prism_training.py --CLIP_model ViT-L/14@336px --task retrieval --setting stereotype_freak_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-L/14@336px --task retrieval --setting stereotype_freak_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-L/14@336px --task retrieval --setting stereotype_freak_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_freak_race
python debiasing_methods/prism_training.py --CLIP_model ViT-L/14@336px --task retrieval --setting stereotype_freak_race --epochs 1 --num_bases 0 --num_samples 200 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-L/14@336px --task retrieval --setting stereotype_freak_race --epochs 1 --num_bases 0 --num_samples 200 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-L/14@336px --task retrieval --setting stereotype_freak_race --epochs 1 --num_bases 0 --num_samples 200 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_freak_utkrace
python debiasing_methods/prism_training.py --CLIP_model ViT-L/14@336px --task retrieval --setting stereotype_freak_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-L/14@336px --task retrieval --setting stereotype_freak_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-L/14@336px --task retrieval --setting stereotype_freak_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_gangster_gender
python debiasing_methods/prism_training.py --CLIP_model ViT-L/14@336px --task retrieval --setting stereotype_gangster_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-L/14@336px --task retrieval --setting stereotype_gangster_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-L/14@336px --task retrieval --setting stereotype_gangster_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_gangster_race
python debiasing_methods/prism_training.py --CLIP_model ViT-L/14@336px --task retrieval --setting stereotype_gangster_race --epochs 1 --num_bases 0 --num_samples 200 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-L/14@336px --task retrieval --setting stereotype_gangster_race --epochs 1 --num_bases 0 --num_samples 200 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-L/14@336px --task retrieval --setting stereotype_gangster_race --epochs 1 --num_bases 0 --num_samples 200 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_gangster_utkrace
python debiasing_methods/prism_training.py --CLIP_model ViT-L/14@336px --task retrieval --setting stereotype_gangster_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-L/14@336px --task retrieval --setting stereotype_gangster_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-L/14@336px --task retrieval --setting stereotype_gangster_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_hacker_gender
python debiasing_methods/prism_training.py --CLIP_model ViT-L/14@336px --task retrieval --setting stereotype_hacker_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-L/14@336px --task retrieval --setting stereotype_hacker_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-L/14@336px --task retrieval --setting stereotype_hacker_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_hacker_race
python debiasing_methods/prism_training.py --CLIP_model ViT-L/14@336px --task retrieval --setting stereotype_hacker_race --epochs 1 --num_bases 0 --num_samples 200 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-L/14@336px --task retrieval --setting stereotype_hacker_race --epochs 1 --num_bases 0 --num_samples 200 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-L/14@336px --task retrieval --setting stereotype_hacker_race --epochs 1 --num_bases 0 --num_samples 200 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_hacker_utkrace
python debiasing_methods/prism_training.py --CLIP_model ViT-L/14@336px --task retrieval --setting stereotype_hacker_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-L/14@336px --task retrieval --setting stereotype_hacker_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-L/14@336px --task retrieval --setting stereotype_hacker_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_illegal-person_gender
python debiasing_methods/prism_training.py --CLIP_model ViT-L/14@336px --task retrieval --setting stereotype_illegal-person_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-L/14@336px --task retrieval --setting stereotype_illegal-person_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-L/14@336px --task retrieval --setting stereotype_illegal-person_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_illegal-person_race
python debiasing_methods/prism_training.py --CLIP_model ViT-L/14@336px --task retrieval --setting stereotype_illegal-person_race --epochs 1 --num_bases 0 --num_samples 200 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-L/14@336px --task retrieval --setting stereotype_illegal-person_race --epochs 1 --num_bases 0 --num_samples 200 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-L/14@336px --task retrieval --setting stereotype_illegal-person_race --epochs 1 --num_bases 0 --num_samples 200 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_illegal-person_utkrace
python debiasing_methods/prism_training.py --CLIP_model ViT-L/14@336px --task retrieval --setting stereotype_illegal-person_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-L/14@336px --task retrieval --setting stereotype_illegal-person_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-L/14@336px --task retrieval --setting stereotype_illegal-person_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_insane-person_gender
python debiasing_methods/prism_training.py --CLIP_model ViT-L/14@336px --task retrieval --setting stereotype_insane-person_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-L/14@336px --task retrieval --setting stereotype_insane-person_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-L/14@336px --task retrieval --setting stereotype_insane-person_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_insane-person_race
python debiasing_methods/prism_training.py --CLIP_model ViT-L/14@336px --task retrieval --setting stereotype_insane-person_race --epochs 1 --num_bases 0 --num_samples 200 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-L/14@336px --task retrieval --setting stereotype_insane-person_race --epochs 1 --num_bases 0 --num_samples 200 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-L/14@336px --task retrieval --setting stereotype_insane-person_race --epochs 1 --num_bases 0 --num_samples 200 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_insane-person_utkrace
python debiasing_methods/prism_training.py --CLIP_model ViT-L/14@336px --task retrieval --setting stereotype_insane-person_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-L/14@336px --task retrieval --setting stereotype_insane-person_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-L/14@336px --task retrieval --setting stereotype_insane-person_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_lawless-person_gender
python debiasing_methods/prism_training.py --CLIP_model ViT-L/14@336px --task retrieval --setting stereotype_lawless-person_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-L/14@336px --task retrieval --setting stereotype_lawless-person_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-L/14@336px --task retrieval --setting stereotype_lawless-person_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_lawless-person_race
python debiasing_methods/prism_training.py --CLIP_model ViT-L/14@336px --task retrieval --setting stereotype_lawless-person_race --epochs 1 --num_bases 0 --num_samples 200 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-L/14@336px --task retrieval --setting stereotype_lawless-person_race --epochs 1 --num_bases 0 --num_samples 200 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-L/14@336px --task retrieval --setting stereotype_lawless-person_race --epochs 1 --num_bases 0 --num_samples 200 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_lawless-person_utkrace
python debiasing_methods/prism_training.py --CLIP_model ViT-L/14@336px --task retrieval --setting stereotype_lawless-person_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-L/14@336px --task retrieval --setting stereotype_lawless-person_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-L/14@336px --task retrieval --setting stereotype_lawless-person_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_mugger_gender
python debiasing_methods/prism_training.py --CLIP_model ViT-L/14@336px --task retrieval --setting stereotype_mugger_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-L/14@336px --task retrieval --setting stereotype_mugger_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-L/14@336px --task retrieval --setting stereotype_mugger_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_mugger_race
python debiasing_methods/prism_training.py --CLIP_model ViT-L/14@336px --task retrieval --setting stereotype_mugger_race --epochs 1 --num_bases 0 --num_samples 200 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-L/14@336px --task retrieval --setting stereotype_mugger_race --epochs 1 --num_bases 0 --num_samples 200 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-L/14@336px --task retrieval --setting stereotype_mugger_race --epochs 1 --num_bases 0 --num_samples 200 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_mugger_utkrace
python debiasing_methods/prism_training.py --CLIP_model ViT-L/14@336px --task retrieval --setting stereotype_mugger_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-L/14@336px --task retrieval --setting stereotype_mugger_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-L/14@336px --task retrieval --setting stereotype_mugger_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_murderer_gender
python debiasing_methods/prism_training.py --CLIP_model ViT-L/14@336px --task retrieval --setting stereotype_murderer_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-L/14@336px --task retrieval --setting stereotype_murderer_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-L/14@336px --task retrieval --setting stereotype_murderer_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_murderer_race
python debiasing_methods/prism_training.py --CLIP_model ViT-L/14@336px --task retrieval --setting stereotype_murderer_race --epochs 1 --num_bases 0 --num_samples 200 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-L/14@336px --task retrieval --setting stereotype_murderer_race --epochs 1 --num_bases 0 --num_samples 200 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-L/14@336px --task retrieval --setting stereotype_murderer_race --epochs 1 --num_bases 0 --num_samples 200 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_murderer_utkrace
python debiasing_methods/prism_training.py --CLIP_model ViT-L/14@336px --task retrieval --setting stereotype_murderer_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-L/14@336px --task retrieval --setting stereotype_murderer_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-L/14@336px --task retrieval --setting stereotype_murderer_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_psychopath_gender
python debiasing_methods/prism_training.py --CLIP_model ViT-L/14@336px --task retrieval --setting stereotype_psychopath_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-L/14@336px --task retrieval --setting stereotype_psychopath_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-L/14@336px --task retrieval --setting stereotype_psychopath_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_psychopath_race
python debiasing_methods/prism_training.py --CLIP_model ViT-L/14@336px --task retrieval --setting stereotype_psychopath_race --epochs 1 --num_bases 0 --num_samples 200 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-L/14@336px --task retrieval --setting stereotype_psychopath_race --epochs 1 --num_bases 0 --num_samples 200 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-L/14@336px --task retrieval --setting stereotype_psychopath_race --epochs 1 --num_bases 0 --num_samples 200 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_psychopath_utkrace
python debiasing_methods/prism_training.py --CLIP_model ViT-L/14@336px --task retrieval --setting stereotype_psychopath_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-L/14@336px --task retrieval --setting stereotype_psychopath_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-L/14@336px --task retrieval --setting stereotype_psychopath_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_robber_gender
python debiasing_methods/prism_training.py --CLIP_model ViT-L/14@336px --task retrieval --setting stereotype_robber_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-L/14@336px --task retrieval --setting stereotype_robber_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-L/14@336px --task retrieval --setting stereotype_robber_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_robber_race
python debiasing_methods/prism_training.py --CLIP_model ViT-L/14@336px --task retrieval --setting stereotype_robber_race --epochs 1 --num_bases 0 --num_samples 200 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-L/14@336px --task retrieval --setting stereotype_robber_race --epochs 1 --num_bases 0 --num_samples 200 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-L/14@336px --task retrieval --setting stereotype_robber_race --epochs 1 --num_bases 0 --num_samples 200 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_robber_utkrace
python debiasing_methods/prism_training.py --CLIP_model ViT-L/14@336px --task retrieval --setting stereotype_robber_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-L/14@336px --task retrieval --setting stereotype_robber_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-L/14@336px --task retrieval --setting stereotype_robber_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_savage_gender
python debiasing_methods/prism_training.py --CLIP_model ViT-L/14@336px --task retrieval --setting stereotype_savage_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-L/14@336px --task retrieval --setting stereotype_savage_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-L/14@336px --task retrieval --setting stereotype_savage_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_savage_race
python debiasing_methods/prism_training.py --CLIP_model ViT-L/14@336px --task retrieval --setting stereotype_savage_race --epochs 1 --num_bases 0 --num_samples 200 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-L/14@336px --task retrieval --setting stereotype_savage_race --epochs 1 --num_bases 0 --num_samples 200 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-L/14@336px --task retrieval --setting stereotype_savage_race --epochs 1 --num_bases 0 --num_samples 200 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_savage_utkrace
python debiasing_methods/prism_training.py --CLIP_model ViT-L/14@336px --task retrieval --setting stereotype_savage_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-L/14@336px --task retrieval --setting stereotype_savage_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-L/14@336px --task retrieval --setting stereotype_savage_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_shoplifter_gender
python debiasing_methods/prism_training.py --CLIP_model ViT-L/14@336px --task retrieval --setting stereotype_shoplifter_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-L/14@336px --task retrieval --setting stereotype_shoplifter_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-L/14@336px --task retrieval --setting stereotype_shoplifter_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_shoplifter_race
python debiasing_methods/prism_training.py --CLIP_model ViT-L/14@336px --task retrieval --setting stereotype_shoplifter_race --epochs 1 --num_bases 0 --num_samples 200 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-L/14@336px --task retrieval --setting stereotype_shoplifter_race --epochs 1 --num_bases 0 --num_samples 200 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-L/14@336px --task retrieval --setting stereotype_shoplifter_race --epochs 1 --num_bases 0 --num_samples 200 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_shoplifter_utkrace
python debiasing_methods/prism_training.py --CLIP_model ViT-L/14@336px --task retrieval --setting stereotype_shoplifter_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-L/14@336px --task retrieval --setting stereotype_shoplifter_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-L/14@336px --task retrieval --setting stereotype_shoplifter_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_terrorist_gender
python debiasing_methods/prism_training.py --CLIP_model ViT-L/14@336px --task retrieval --setting stereotype_terrorist_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-L/14@336px --task retrieval --setting stereotype_terrorist_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-L/14@336px --task retrieval --setting stereotype_terrorist_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_terrorist_race
python debiasing_methods/prism_training.py --CLIP_model ViT-L/14@336px --task retrieval --setting stereotype_terrorist_race --epochs 1 --num_bases 0 --num_samples 200 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-L/14@336px --task retrieval --setting stereotype_terrorist_race --epochs 1 --num_bases 0 --num_samples 200 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-L/14@336px --task retrieval --setting stereotype_terrorist_race --epochs 1 --num_bases 0 --num_samples 200 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_terrorist_utkrace
python debiasing_methods/prism_training.py --CLIP_model ViT-L/14@336px --task retrieval --setting stereotype_terrorist_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-L/14@336px --task retrieval --setting stereotype_terrorist_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-L/14@336px --task retrieval --setting stereotype_terrorist_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_thief_gender
python debiasing_methods/prism_training.py --CLIP_model ViT-L/14@336px --task retrieval --setting stereotype_thief_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-L/14@336px --task retrieval --setting stereotype_thief_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-L/14@336px --task retrieval --setting stereotype_thief_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_thief_race
python debiasing_methods/prism_training.py --CLIP_model ViT-L/14@336px --task retrieval --setting stereotype_thief_race --epochs 1 --num_bases 0 --num_samples 200 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-L/14@336px --task retrieval --setting stereotype_thief_race --epochs 1 --num_bases 0 --num_samples 200 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-L/14@336px --task retrieval --setting stereotype_thief_race --epochs 1 --num_bases 0 --num_samples 200 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_thief_utkrace
python debiasing_methods/prism_training.py --CLIP_model ViT-L/14@336px --task retrieval --setting stereotype_thief_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-L/14@336px --task retrieval --setting stereotype_thief_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-L/14@336px --task retrieval --setting stereotype_thief_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_thug_gender
python debiasing_methods/prism_training.py --CLIP_model ViT-L/14@336px --task retrieval --setting stereotype_thug_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-L/14@336px --task retrieval --setting stereotype_thug_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-L/14@336px --task retrieval --setting stereotype_thug_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_thug_race
python debiasing_methods/prism_training.py --CLIP_model ViT-L/14@336px --task retrieval --setting stereotype_thug_race --epochs 1 --num_bases 0 --num_samples 200 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-L/14@336px --task retrieval --setting stereotype_thug_race --epochs 1 --num_bases 0 --num_samples 200 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-L/14@336px --task retrieval --setting stereotype_thug_race --epochs 1 --num_bases 0 --num_samples 200 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_thug_utkrace
python debiasing_methods/prism_training.py --CLIP_model ViT-L/14@336px --task retrieval --setting stereotype_thug_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-L/14@336px --task retrieval --setting stereotype_thug_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-L/14@336px --task retrieval --setting stereotype_thug_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_violent-person_gender
python debiasing_methods/prism_training.py --CLIP_model ViT-L/14@336px --task retrieval --setting stereotype_violent-person_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-L/14@336px --task retrieval --setting stereotype_violent-person_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-L/14@336px --task retrieval --setting stereotype_violent-person_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_violent-person_race
python debiasing_methods/prism_training.py --CLIP_model ViT-L/14@336px --task retrieval --setting stereotype_violent-person_race --epochs 1 --num_bases 0 --num_samples 200 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-L/14@336px --task retrieval --setting stereotype_violent-person_race --epochs 1 --num_bases 0 --num_samples 200 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-L/14@336px --task retrieval --setting stereotype_violent-person_race --epochs 1 --num_bases 0 --num_samples 200 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_violent-person_utkrace
python debiasing_methods/prism_training.py --CLIP_model ViT-L/14@336px --task retrieval --setting stereotype_violent-person_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-L/14@336px --task retrieval --setting stereotype_violent-person_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-L/14@336px --task retrieval --setting stereotype_violent-person_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: celeba_black_hair_gender
python debiasing_methods/prism_training.py --CLIP_model ViT-L/14@336px --task retrieval --setting celeba_black_hair_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-L/14@336px --task retrieval --setting celeba_black_hair_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-L/14@336px --task retrieval --setting celeba_black_hair_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: celeba_blond_hair_gender
python debiasing_methods/prism_training.py --CLIP_model ViT-L/14@336px --task retrieval --setting celeba_blond_hair_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-L/14@336px --task retrieval --setting celeba_blond_hair_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-L/14@336px --task retrieval --setting celeba_blond_hair_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: celeba_brown_hair_gender
python debiasing_methods/prism_training.py --CLIP_model ViT-L/14@336px --task retrieval --setting celeba_brown_hair_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-L/14@336px --task retrieval --setting celeba_brown_hair_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-L/14@336px --task retrieval --setting celeba_brown_hair_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: celeba_gray_hair_gender
python debiasing_methods/prism_training.py --CLIP_model ViT-L/14@336px --task retrieval --setting celeba_gray_hair_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-L/14@336px --task retrieval --setting celeba_gray_hair_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-L/14@336px --task retrieval --setting celeba_gray_hair_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random



# Zero-shot classification tasks
## Backbone: ViT-B/16
### Setting: celeba_blond_hair_gender
python debiasing_methods/prism_training.py --CLIP_model ViT-B/16 --task zs_classification --setting celeba_blond_hair_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-B/16 --task zs_classification --setting celeba_blond_hair_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-B/16 --task zs_classification --setting celeba_blond_hair_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: waterbirds
python debiasing_methods/prism_training.py --CLIP_model ViT-B/16 --task zs_classification --setting waterbirds --epochs 1 --num_bases 0 --num_samples 500 --seed 0 --lr 0.1 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-B/16 --task zs_classification --setting waterbirds --epochs 1 --num_bases 0 --num_samples 500 --seed 1 --lr 0.1 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-B/16 --task zs_classification --setting waterbirds --epochs 1 --num_bases 0 --num_samples 500 --seed 2 --lr 0.1 --batch_size 64 --wd 0 --init_weight random


## Backbone: ViT-L/14@336px
### Setting: celeba_blond_hair_gender
python debiasing_methods/prism_training.py --CLIP_model ViT-L/14@336px --task zs_classification --setting celeba_blond_hair_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-L/14@336px --task zs_classification --setting celeba_blond_hair_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-L/14@336px --task zs_classification --setting celeba_blond_hair_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: waterbirds
python debiasing_methods/prism_training.py --CLIP_model ViT-L/14@336px --task zs_classification --setting waterbirds --epochs 1 --num_bases 0 --num_samples 500 --seed 0 --lr 0.1 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-L/14@336px --task zs_classification --setting waterbirds --epochs 1 --num_bases 0 --num_samples 500 --seed 1 --lr 0.1 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model ViT-L/14@336px --task zs_classification --setting waterbirds --epochs 1 --num_bases 0 --num_samples 500 --seed 2 --lr 0.1 --batch_size 64 --wd 0 --init_weight random


# --- ResNet backbones (supp. mat.) --- #


# Retrieval tasks
## Backbone: RN50
### Setting: stereotype_addict_gender
python debiasing_methods/prism_training.py --CLIP_model RN50 --task retrieval --setting stereotype_addict_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN50 --task retrieval --setting stereotype_addict_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN50 --task retrieval --setting stereotype_addict_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_addict_race
python debiasing_methods/prism_training.py --CLIP_model RN50 --task retrieval --setting stereotype_addict_race --epochs 1 --num_bases 0 --num_samples 200 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN50 --task retrieval --setting stereotype_addict_race --epochs 1 --num_bases 0 --num_samples 200 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN50 --task retrieval --setting stereotype_addict_race --epochs 1 --num_bases 0 --num_samples 200 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_addict_utkrace
python debiasing_methods/prism_training.py --CLIP_model RN50 --task retrieval --setting stereotype_addict_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN50 --task retrieval --setting stereotype_addict_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN50 --task retrieval --setting stereotype_addict_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_burglar_gender
python debiasing_methods/prism_training.py --CLIP_model RN50 --task retrieval --setting stereotype_burglar_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN50 --task retrieval --setting stereotype_burglar_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN50 --task retrieval --setting stereotype_burglar_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_burglar_race
python debiasing_methods/prism_training.py --CLIP_model RN50 --task retrieval --setting stereotype_burglar_race --epochs 1 --num_bases 0 --num_samples 200 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN50 --task retrieval --setting stereotype_burglar_race --epochs 1 --num_bases 0 --num_samples 200 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN50 --task retrieval --setting stereotype_burglar_race --epochs 1 --num_bases 0 --num_samples 200 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_burglar_utkrace
python debiasing_methods/prism_training.py --CLIP_model RN50 --task retrieval --setting stereotype_burglar_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN50 --task retrieval --setting stereotype_burglar_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN50 --task retrieval --setting stereotype_burglar_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_criminal_gender
python debiasing_methods/prism_training.py --CLIP_model RN50 --task retrieval --setting stereotype_criminal_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN50 --task retrieval --setting stereotype_criminal_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN50 --task retrieval --setting stereotype_criminal_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_criminal_race
python debiasing_methods/prism_training.py --CLIP_model RN50 --task retrieval --setting stereotype_criminal_race --epochs 1 --num_bases 0 --num_samples 200 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN50 --task retrieval --setting stereotype_criminal_race --epochs 1 --num_bases 0 --num_samples 200 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN50 --task retrieval --setting stereotype_criminal_race --epochs 1 --num_bases 0 --num_samples 200 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_criminal_utkrace
python debiasing_methods/prism_training.py --CLIP_model RN50 --task retrieval --setting stereotype_criminal_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN50 --task retrieval --setting stereotype_criminal_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN50 --task retrieval --setting stereotype_criminal_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_dangerous-person_gender
python debiasing_methods/prism_training.py --CLIP_model RN50 --task retrieval --setting stereotype_dangerous-person_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN50 --task retrieval --setting stereotype_dangerous-person_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN50 --task retrieval --setting stereotype_dangerous-person_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_dangerous-person_race
python debiasing_methods/prism_training.py --CLIP_model RN50 --task retrieval --setting stereotype_dangerous-person_race --epochs 1 --num_bases 0 --num_samples 200 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN50 --task retrieval --setting stereotype_dangerous-person_race --epochs 1 --num_bases 0 --num_samples 200 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN50 --task retrieval --setting stereotype_dangerous-person_race --epochs 1 --num_bases 0 --num_samples 200 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_dangerous-person_utkrace
python debiasing_methods/prism_training.py --CLIP_model RN50 --task retrieval --setting stereotype_dangerous-person_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN50 --task retrieval --setting stereotype_dangerous-person_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN50 --task retrieval --setting stereotype_dangerous-person_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_delinquent_gender
python debiasing_methods/prism_training.py --CLIP_model RN50 --task retrieval --setting stereotype_delinquent_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN50 --task retrieval --setting stereotype_delinquent_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN50 --task retrieval --setting stereotype_delinquent_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_delinquent_race
python debiasing_methods/prism_training.py --CLIP_model RN50 --task retrieval --setting stereotype_delinquent_race --epochs 1 --num_bases 0 --num_samples 200 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN50 --task retrieval --setting stereotype_delinquent_race --epochs 1 --num_bases 0 --num_samples 200 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN50 --task retrieval --setting stereotype_delinquent_race --epochs 1 --num_bases 0 --num_samples 200 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_delinquent_utkrace
python debiasing_methods/prism_training.py --CLIP_model RN50 --task retrieval --setting stereotype_delinquent_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN50 --task retrieval --setting stereotype_delinquent_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN50 --task retrieval --setting stereotype_delinquent_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_deviant_gender
python debiasing_methods/prism_training.py --CLIP_model RN50 --task retrieval --setting stereotype_deviant_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN50 --task retrieval --setting stereotype_deviant_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN50 --task retrieval --setting stereotype_deviant_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_deviant_race
python debiasing_methods/prism_training.py --CLIP_model RN50 --task retrieval --setting stereotype_deviant_race --epochs 1 --num_bases 0 --num_samples 200 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN50 --task retrieval --setting stereotype_deviant_race --epochs 1 --num_bases 0 --num_samples 200 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN50 --task retrieval --setting stereotype_deviant_race --epochs 1 --num_bases 0 --num_samples 200 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_deviant_utkrace
python debiasing_methods/prism_training.py --CLIP_model RN50 --task retrieval --setting stereotype_deviant_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN50 --task retrieval --setting stereotype_deviant_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN50 --task retrieval --setting stereotype_deviant_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_embezzler_gender
python debiasing_methods/prism_training.py --CLIP_model RN50 --task retrieval --setting stereotype_embezzler_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN50 --task retrieval --setting stereotype_embezzler_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN50 --task retrieval --setting stereotype_embezzler_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_embezzler_race
python debiasing_methods/prism_training.py --CLIP_model RN50 --task retrieval --setting stereotype_embezzler_race --epochs 1 --num_bases 0 --num_samples 200 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN50 --task retrieval --setting stereotype_embezzler_race --epochs 1 --num_bases 0 --num_samples 200 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN50 --task retrieval --setting stereotype_embezzler_race --epochs 1 --num_bases 0 --num_samples 200 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_embezzler_utkrace
python debiasing_methods/prism_training.py --CLIP_model RN50 --task retrieval --setting stereotype_embezzler_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN50 --task retrieval --setting stereotype_embezzler_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN50 --task retrieval --setting stereotype_embezzler_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_felon_gender
python debiasing_methods/prism_training.py --CLIP_model RN50 --task retrieval --setting stereotype_felon_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN50 --task retrieval --setting stereotype_felon_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN50 --task retrieval --setting stereotype_felon_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_felon_race
python debiasing_methods/prism_training.py --CLIP_model RN50 --task retrieval --setting stereotype_felon_race --epochs 1 --num_bases 0 --num_samples 200 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN50 --task retrieval --setting stereotype_felon_race --epochs 1 --num_bases 0 --num_samples 200 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN50 --task retrieval --setting stereotype_felon_race --epochs 1 --num_bases 0 --num_samples 200 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_felon_utkrace
python debiasing_methods/prism_training.py --CLIP_model RN50 --task retrieval --setting stereotype_felon_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN50 --task retrieval --setting stereotype_felon_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN50 --task retrieval --setting stereotype_felon_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_fraud_gender
python debiasing_methods/prism_training.py --CLIP_model RN50 --task retrieval --setting stereotype_fraud_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN50 --task retrieval --setting stereotype_fraud_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN50 --task retrieval --setting stereotype_fraud_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_fraud_race
python debiasing_methods/prism_training.py --CLIP_model RN50 --task retrieval --setting stereotype_fraud_race --epochs 1 --num_bases 0 --num_samples 200 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN50 --task retrieval --setting stereotype_fraud_race --epochs 1 --num_bases 0 --num_samples 200 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN50 --task retrieval --setting stereotype_fraud_race --epochs 1 --num_bases 0 --num_samples 200 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_fraud_utkrace
python debiasing_methods/prism_training.py --CLIP_model RN50 --task retrieval --setting stereotype_fraud_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN50 --task retrieval --setting stereotype_fraud_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN50 --task retrieval --setting stereotype_fraud_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_freak_gender
python debiasing_methods/prism_training.py --CLIP_model RN50 --task retrieval --setting stereotype_freak_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN50 --task retrieval --setting stereotype_freak_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN50 --task retrieval --setting stereotype_freak_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_freak_race
python debiasing_methods/prism_training.py --CLIP_model RN50 --task retrieval --setting stereotype_freak_race --epochs 1 --num_bases 0 --num_samples 200 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN50 --task retrieval --setting stereotype_freak_race --epochs 1 --num_bases 0 --num_samples 200 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN50 --task retrieval --setting stereotype_freak_race --epochs 1 --num_bases 0 --num_samples 200 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_freak_utkrace
python debiasing_methods/prism_training.py --CLIP_model RN50 --task retrieval --setting stereotype_freak_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN50 --task retrieval --setting stereotype_freak_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN50 --task retrieval --setting stereotype_freak_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_gangster_gender
python debiasing_methods/prism_training.py --CLIP_model RN50 --task retrieval --setting stereotype_gangster_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN50 --task retrieval --setting stereotype_gangster_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN50 --task retrieval --setting stereotype_gangster_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_gangster_race
python debiasing_methods/prism_training.py --CLIP_model RN50 --task retrieval --setting stereotype_gangster_race --epochs 1 --num_bases 0 --num_samples 200 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN50 --task retrieval --setting stereotype_gangster_race --epochs 1 --num_bases 0 --num_samples 200 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN50 --task retrieval --setting stereotype_gangster_race --epochs 1 --num_bases 0 --num_samples 200 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_gangster_utkrace
python debiasing_methods/prism_training.py --CLIP_model RN50 --task retrieval --setting stereotype_gangster_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN50 --task retrieval --setting stereotype_gangster_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN50 --task retrieval --setting stereotype_gangster_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_hacker_gender
python debiasing_methods/prism_training.py --CLIP_model RN50 --task retrieval --setting stereotype_hacker_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN50 --task retrieval --setting stereotype_hacker_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN50 --task retrieval --setting stereotype_hacker_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_hacker_race
python debiasing_methods/prism_training.py --CLIP_model RN50 --task retrieval --setting stereotype_hacker_race --epochs 1 --num_bases 0 --num_samples 200 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN50 --task retrieval --setting stereotype_hacker_race --epochs 1 --num_bases 0 --num_samples 200 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN50 --task retrieval --setting stereotype_hacker_race --epochs 1 --num_bases 0 --num_samples 200 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_hacker_utkrace
python debiasing_methods/prism_training.py --CLIP_model RN50 --task retrieval --setting stereotype_hacker_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN50 --task retrieval --setting stereotype_hacker_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN50 --task retrieval --setting stereotype_hacker_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_illegal-person_gender
python debiasing_methods/prism_training.py --CLIP_model RN50 --task retrieval --setting stereotype_illegal-person_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN50 --task retrieval --setting stereotype_illegal-person_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN50 --task retrieval --setting stereotype_illegal-person_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_illegal-person_race
python debiasing_methods/prism_training.py --CLIP_model RN50 --task retrieval --setting stereotype_illegal-person_race --epochs 1 --num_bases 0 --num_samples 200 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN50 --task retrieval --setting stereotype_illegal-person_race --epochs 1 --num_bases 0 --num_samples 200 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN50 --task retrieval --setting stereotype_illegal-person_race --epochs 1 --num_bases 0 --num_samples 200 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_illegal-person_utkrace
python debiasing_methods/prism_training.py --CLIP_model RN50 --task retrieval --setting stereotype_illegal-person_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN50 --task retrieval --setting stereotype_illegal-person_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN50 --task retrieval --setting stereotype_illegal-person_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_insane-person_gender
python debiasing_methods/prism_training.py --CLIP_model RN50 --task retrieval --setting stereotype_insane-person_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN50 --task retrieval --setting stereotype_insane-person_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN50 --task retrieval --setting stereotype_insane-person_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_insane-person_race
python debiasing_methods/prism_training.py --CLIP_model RN50 --task retrieval --setting stereotype_insane-person_race --epochs 1 --num_bases 0 --num_samples 200 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN50 --task retrieval --setting stereotype_insane-person_race --epochs 1 --num_bases 0 --num_samples 200 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN50 --task retrieval --setting stereotype_insane-person_race --epochs 1 --num_bases 0 --num_samples 200 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_insane-person_utkrace
python debiasing_methods/prism_training.py --CLIP_model RN50 --task retrieval --setting stereotype_insane-person_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN50 --task retrieval --setting stereotype_insane-person_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN50 --task retrieval --setting stereotype_insane-person_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_lawless-person_gender
python debiasing_methods/prism_training.py --CLIP_model RN50 --task retrieval --setting stereotype_lawless-person_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN50 --task retrieval --setting stereotype_lawless-person_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN50 --task retrieval --setting stereotype_lawless-person_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_lawless-person_race
python debiasing_methods/prism_training.py --CLIP_model RN50 --task retrieval --setting stereotype_lawless-person_race --epochs 1 --num_bases 0 --num_samples 200 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN50 --task retrieval --setting stereotype_lawless-person_race --epochs 1 --num_bases 0 --num_samples 200 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN50 --task retrieval --setting stereotype_lawless-person_race --epochs 1 --num_bases 0 --num_samples 200 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_lawless-person_utkrace
python debiasing_methods/prism_training.py --CLIP_model RN50 --task retrieval --setting stereotype_lawless-person_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN50 --task retrieval --setting stereotype_lawless-person_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN50 --task retrieval --setting stereotype_lawless-person_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_mugger_gender
python debiasing_methods/prism_training.py --CLIP_model RN50 --task retrieval --setting stereotype_mugger_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN50 --task retrieval --setting stereotype_mugger_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN50 --task retrieval --setting stereotype_mugger_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_mugger_race
python debiasing_methods/prism_training.py --CLIP_model RN50 --task retrieval --setting stereotype_mugger_race --epochs 1 --num_bases 0 --num_samples 200 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN50 --task retrieval --setting stereotype_mugger_race --epochs 1 --num_bases 0 --num_samples 200 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN50 --task retrieval --setting stereotype_mugger_race --epochs 1 --num_bases 0 --num_samples 200 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_mugger_utkrace
python debiasing_methods/prism_training.py --CLIP_model RN50 --task retrieval --setting stereotype_mugger_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN50 --task retrieval --setting stereotype_mugger_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN50 --task retrieval --setting stereotype_mugger_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_murderer_gender
python debiasing_methods/prism_training.py --CLIP_model RN50 --task retrieval --setting stereotype_murderer_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN50 --task retrieval --setting stereotype_murderer_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN50 --task retrieval --setting stereotype_murderer_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_murderer_race
python debiasing_methods/prism_training.py --CLIP_model RN50 --task retrieval --setting stereotype_murderer_race --epochs 1 --num_bases 0 --num_samples 200 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN50 --task retrieval --setting stereotype_murderer_race --epochs 1 --num_bases 0 --num_samples 200 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN50 --task retrieval --setting stereotype_murderer_race --epochs 1 --num_bases 0 --num_samples 200 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_murderer_utkrace
python debiasing_methods/prism_training.py --CLIP_model RN50 --task retrieval --setting stereotype_murderer_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN50 --task retrieval --setting stereotype_murderer_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN50 --task retrieval --setting stereotype_murderer_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_psychopath_gender
python debiasing_methods/prism_training.py --CLIP_model RN50 --task retrieval --setting stereotype_psychopath_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN50 --task retrieval --setting stereotype_psychopath_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN50 --task retrieval --setting stereotype_psychopath_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_psychopath_race
python debiasing_methods/prism_training.py --CLIP_model RN50 --task retrieval --setting stereotype_psychopath_race --epochs 1 --num_bases 0 --num_samples 200 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN50 --task retrieval --setting stereotype_psychopath_race --epochs 1 --num_bases 0 --num_samples 200 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN50 --task retrieval --setting stereotype_psychopath_race --epochs 1 --num_bases 0 --num_samples 200 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_psychopath_utkrace
python debiasing_methods/prism_training.py --CLIP_model RN50 --task retrieval --setting stereotype_psychopath_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN50 --task retrieval --setting stereotype_psychopath_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN50 --task retrieval --setting stereotype_psychopath_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_robber_gender
python debiasing_methods/prism_training.py --CLIP_model RN50 --task retrieval --setting stereotype_robber_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN50 --task retrieval --setting stereotype_robber_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN50 --task retrieval --setting stereotype_robber_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_robber_race
python debiasing_methods/prism_training.py --CLIP_model RN50 --task retrieval --setting stereotype_robber_race --epochs 1 --num_bases 0 --num_samples 200 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN50 --task retrieval --setting stereotype_robber_race --epochs 1 --num_bases 0 --num_samples 200 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN50 --task retrieval --setting stereotype_robber_race --epochs 1 --num_bases 0 --num_samples 200 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_robber_utkrace
python debiasing_methods/prism_training.py --CLIP_model RN50 --task retrieval --setting stereotype_robber_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN50 --task retrieval --setting stereotype_robber_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN50 --task retrieval --setting stereotype_robber_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_savage_gender
python debiasing_methods/prism_training.py --CLIP_model RN50 --task retrieval --setting stereotype_savage_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN50 --task retrieval --setting stereotype_savage_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN50 --task retrieval --setting stereotype_savage_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_savage_race
python debiasing_methods/prism_training.py --CLIP_model RN50 --task retrieval --setting stereotype_savage_race --epochs 1 --num_bases 0 --num_samples 200 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN50 --task retrieval --setting stereotype_savage_race --epochs 1 --num_bases 0 --num_samples 200 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN50 --task retrieval --setting stereotype_savage_race --epochs 1 --num_bases 0 --num_samples 200 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_savage_utkrace
python debiasing_methods/prism_training.py --CLIP_model RN50 --task retrieval --setting stereotype_savage_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN50 --task retrieval --setting stereotype_savage_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN50 --task retrieval --setting stereotype_savage_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_shoplifter_gender
python debiasing_methods/prism_training.py --CLIP_model RN50 --task retrieval --setting stereotype_shoplifter_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN50 --task retrieval --setting stereotype_shoplifter_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN50 --task retrieval --setting stereotype_shoplifter_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_shoplifter_race
python debiasing_methods/prism_training.py --CLIP_model RN50 --task retrieval --setting stereotype_shoplifter_race --epochs 1 --num_bases 0 --num_samples 200 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN50 --task retrieval --setting stereotype_shoplifter_race --epochs 1 --num_bases 0 --num_samples 200 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN50 --task retrieval --setting stereotype_shoplifter_race --epochs 1 --num_bases 0 --num_samples 200 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_shoplifter_utkrace
python debiasing_methods/prism_training.py --CLIP_model RN50 --task retrieval --setting stereotype_shoplifter_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN50 --task retrieval --setting stereotype_shoplifter_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN50 --task retrieval --setting stereotype_shoplifter_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_terrorist_gender
python debiasing_methods/prism_training.py --CLIP_model RN50 --task retrieval --setting stereotype_terrorist_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN50 --task retrieval --setting stereotype_terrorist_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN50 --task retrieval --setting stereotype_terrorist_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_terrorist_race
python debiasing_methods/prism_training.py --CLIP_model RN50 --task retrieval --setting stereotype_terrorist_race --epochs 1 --num_bases 0 --num_samples 200 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN50 --task retrieval --setting stereotype_terrorist_race --epochs 1 --num_bases 0 --num_samples 200 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN50 --task retrieval --setting stereotype_terrorist_race --epochs 1 --num_bases 0 --num_samples 200 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_terrorist_utkrace
python debiasing_methods/prism_training.py --CLIP_model RN50 --task retrieval --setting stereotype_terrorist_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN50 --task retrieval --setting stereotype_terrorist_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN50 --task retrieval --setting stereotype_terrorist_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_thief_gender
python debiasing_methods/prism_training.py --CLIP_model RN50 --task retrieval --setting stereotype_thief_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN50 --task retrieval --setting stereotype_thief_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN50 --task retrieval --setting stereotype_thief_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_thief_race
python debiasing_methods/prism_training.py --CLIP_model RN50 --task retrieval --setting stereotype_thief_race --epochs 1 --num_bases 0 --num_samples 200 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN50 --task retrieval --setting stereotype_thief_race --epochs 1 --num_bases 0 --num_samples 200 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN50 --task retrieval --setting stereotype_thief_race --epochs 1 --num_bases 0 --num_samples 200 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_thief_utkrace
python debiasing_methods/prism_training.py --CLIP_model RN50 --task retrieval --setting stereotype_thief_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN50 --task retrieval --setting stereotype_thief_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN50 --task retrieval --setting stereotype_thief_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_thug_gender
python debiasing_methods/prism_training.py --CLIP_model RN50 --task retrieval --setting stereotype_thug_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN50 --task retrieval --setting stereotype_thug_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN50 --task retrieval --setting stereotype_thug_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_thug_race
python debiasing_methods/prism_training.py --CLIP_model RN50 --task retrieval --setting stereotype_thug_race --epochs 1 --num_bases 0 --num_samples 200 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN50 --task retrieval --setting stereotype_thug_race --epochs 1 --num_bases 0 --num_samples 200 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN50 --task retrieval --setting stereotype_thug_race --epochs 1 --num_bases 0 --num_samples 200 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_thug_utkrace
python debiasing_methods/prism_training.py --CLIP_model RN50 --task retrieval --setting stereotype_thug_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN50 --task retrieval --setting stereotype_thug_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN50 --task retrieval --setting stereotype_thug_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_violent-person_gender
python debiasing_methods/prism_training.py --CLIP_model RN50 --task retrieval --setting stereotype_violent-person_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN50 --task retrieval --setting stereotype_violent-person_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN50 --task retrieval --setting stereotype_violent-person_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_violent-person_race
python debiasing_methods/prism_training.py --CLIP_model RN50 --task retrieval --setting stereotype_violent-person_race --epochs 1 --num_bases 0 --num_samples 200 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN50 --task retrieval --setting stereotype_violent-person_race --epochs 1 --num_bases 0 --num_samples 200 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN50 --task retrieval --setting stereotype_violent-person_race --epochs 1 --num_bases 0 --num_samples 200 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_violent-person_utkrace
python debiasing_methods/prism_training.py --CLIP_model RN50 --task retrieval --setting stereotype_violent-person_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN50 --task retrieval --setting stereotype_violent-person_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN50 --task retrieval --setting stereotype_violent-person_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: celeba_black_hair_gender
python debiasing_methods/prism_training.py --CLIP_model RN50 --task retrieval --setting celeba_black_hair_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN50 --task retrieval --setting celeba_black_hair_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN50 --task retrieval --setting celeba_black_hair_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: celeba_blond_hair_gender
python debiasing_methods/prism_training.py --CLIP_model RN50 --task retrieval --setting celeba_blond_hair_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN50 --task retrieval --setting celeba_blond_hair_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN50 --task retrieval --setting celeba_blond_hair_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: celeba_brown_hair_gender
python debiasing_methods/prism_training.py --CLIP_model RN50 --task retrieval --setting celeba_brown_hair_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN50 --task retrieval --setting celeba_brown_hair_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN50 --task retrieval --setting celeba_brown_hair_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: celeba_gray_hair_gender
python debiasing_methods/prism_training.py --CLIP_model RN50 --task retrieval --setting celeba_gray_hair_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN50 --task retrieval --setting celeba_gray_hair_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN50 --task retrieval --setting celeba_gray_hair_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random


## Backbone: RN101
### Setting: stereotype_addict_gender
python debiasing_methods/prism_training.py --CLIP_model RN101 --task retrieval --setting stereotype_addict_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN101 --task retrieval --setting stereotype_addict_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN101 --task retrieval --setting stereotype_addict_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_addict_race
python debiasing_methods/prism_training.py --CLIP_model RN101 --task retrieval --setting stereotype_addict_race --epochs 1 --num_bases 0 --num_samples 200 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN101 --task retrieval --setting stereotype_addict_race --epochs 1 --num_bases 0 --num_samples 200 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN101 --task retrieval --setting stereotype_addict_race --epochs 1 --num_bases 0 --num_samples 200 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_addict_utkrace
python debiasing_methods/prism_training.py --CLIP_model RN101 --task retrieval --setting stereotype_addict_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN101 --task retrieval --setting stereotype_addict_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN101 --task retrieval --setting stereotype_addict_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_burglar_gender
python debiasing_methods/prism_training.py --CLIP_model RN101 --task retrieval --setting stereotype_burglar_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN101 --task retrieval --setting stereotype_burglar_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN101 --task retrieval --setting stereotype_burglar_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_burglar_race
python debiasing_methods/prism_training.py --CLIP_model RN101 --task retrieval --setting stereotype_burglar_race --epochs 1 --num_bases 0 --num_samples 200 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN101 --task retrieval --setting stereotype_burglar_race --epochs 1 --num_bases 0 --num_samples 200 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN101 --task retrieval --setting stereotype_burglar_race --epochs 1 --num_bases 0 --num_samples 200 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_burglar_utkrace
python debiasing_methods/prism_training.py --CLIP_model RN101 --task retrieval --setting stereotype_burglar_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN101 --task retrieval --setting stereotype_burglar_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN101 --task retrieval --setting stereotype_burglar_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_criminal_gender
python debiasing_methods/prism_training.py --CLIP_model RN101 --task retrieval --setting stereotype_criminal_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN101 --task retrieval --setting stereotype_criminal_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN101 --task retrieval --setting stereotype_criminal_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_criminal_race
python debiasing_methods/prism_training.py --CLIP_model RN101 --task retrieval --setting stereotype_criminal_race --epochs 1 --num_bases 0 --num_samples 200 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN101 --task retrieval --setting stereotype_criminal_race --epochs 1 --num_bases 0 --num_samples 200 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN101 --task retrieval --setting stereotype_criminal_race --epochs 1 --num_bases 0 --num_samples 200 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_criminal_utkrace
python debiasing_methods/prism_training.py --CLIP_model RN101 --task retrieval --setting stereotype_criminal_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN101 --task retrieval --setting stereotype_criminal_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN101 --task retrieval --setting stereotype_criminal_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_dangerous-person_gender
python debiasing_methods/prism_training.py --CLIP_model RN101 --task retrieval --setting stereotype_dangerous-person_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN101 --task retrieval --setting stereotype_dangerous-person_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN101 --task retrieval --setting stereotype_dangerous-person_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_dangerous-person_race
python debiasing_methods/prism_training.py --CLIP_model RN101 --task retrieval --setting stereotype_dangerous-person_race --epochs 1 --num_bases 0 --num_samples 200 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN101 --task retrieval --setting stereotype_dangerous-person_race --epochs 1 --num_bases 0 --num_samples 200 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN101 --task retrieval --setting stereotype_dangerous-person_race --epochs 1 --num_bases 0 --num_samples 200 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_dangerous-person_utkrace
python debiasing_methods/prism_training.py --CLIP_model RN101 --task retrieval --setting stereotype_dangerous-person_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN101 --task retrieval --setting stereotype_dangerous-person_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN101 --task retrieval --setting stereotype_dangerous-person_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_delinquent_gender
python debiasing_methods/prism_training.py --CLIP_model RN101 --task retrieval --setting stereotype_delinquent_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN101 --task retrieval --setting stereotype_delinquent_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN101 --task retrieval --setting stereotype_delinquent_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_delinquent_race
python debiasing_methods/prism_training.py --CLIP_model RN101 --task retrieval --setting stereotype_delinquent_race --epochs 1 --num_bases 0 --num_samples 200 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN101 --task retrieval --setting stereotype_delinquent_race --epochs 1 --num_bases 0 --num_samples 200 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN101 --task retrieval --setting stereotype_delinquent_race --epochs 1 --num_bases 0 --num_samples 200 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_delinquent_utkrace
python debiasing_methods/prism_training.py --CLIP_model RN101 --task retrieval --setting stereotype_delinquent_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN101 --task retrieval --setting stereotype_delinquent_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN101 --task retrieval --setting stereotype_delinquent_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_deviant_gender
python debiasing_methods/prism_training.py --CLIP_model RN101 --task retrieval --setting stereotype_deviant_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN101 --task retrieval --setting stereotype_deviant_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN101 --task retrieval --setting stereotype_deviant_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_deviant_race
python debiasing_methods/prism_training.py --CLIP_model RN101 --task retrieval --setting stereotype_deviant_race --epochs 1 --num_bases 0 --num_samples 200 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN101 --task retrieval --setting stereotype_deviant_race --epochs 1 --num_bases 0 --num_samples 200 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN101 --task retrieval --setting stereotype_deviant_race --epochs 1 --num_bases 0 --num_samples 200 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_deviant_utkrace
python debiasing_methods/prism_training.py --CLIP_model RN101 --task retrieval --setting stereotype_deviant_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN101 --task retrieval --setting stereotype_deviant_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN101 --task retrieval --setting stereotype_deviant_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_embezzler_gender
python debiasing_methods/prism_training.py --CLIP_model RN101 --task retrieval --setting stereotype_embezzler_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN101 --task retrieval --setting stereotype_embezzler_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN101 --task retrieval --setting stereotype_embezzler_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_embezzler_race
python debiasing_methods/prism_training.py --CLIP_model RN101 --task retrieval --setting stereotype_embezzler_race --epochs 1 --num_bases 0 --num_samples 200 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN101 --task retrieval --setting stereotype_embezzler_race --epochs 1 --num_bases 0 --num_samples 200 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN101 --task retrieval --setting stereotype_embezzler_race --epochs 1 --num_bases 0 --num_samples 200 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_embezzler_utkrace
python debiasing_methods/prism_training.py --CLIP_model RN101 --task retrieval --setting stereotype_embezzler_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN101 --task retrieval --setting stereotype_embezzler_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN101 --task retrieval --setting stereotype_embezzler_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_felon_gender
python debiasing_methods/prism_training.py --CLIP_model RN101 --task retrieval --setting stereotype_felon_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN101 --task retrieval --setting stereotype_felon_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN101 --task retrieval --setting stereotype_felon_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_felon_race
python debiasing_methods/prism_training.py --CLIP_model RN101 --task retrieval --setting stereotype_felon_race --epochs 1 --num_bases 0 --num_samples 200 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN101 --task retrieval --setting stereotype_felon_race --epochs 1 --num_bases 0 --num_samples 200 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN101 --task retrieval --setting stereotype_felon_race --epochs 1 --num_bases 0 --num_samples 200 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_felon_utkrace
python debiasing_methods/prism_training.py --CLIP_model RN101 --task retrieval --setting stereotype_felon_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN101 --task retrieval --setting stereotype_felon_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN101 --task retrieval --setting stereotype_felon_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_fraud_gender
python debiasing_methods/prism_training.py --CLIP_model RN101 --task retrieval --setting stereotype_fraud_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN101 --task retrieval --setting stereotype_fraud_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN101 --task retrieval --setting stereotype_fraud_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_fraud_race
python debiasing_methods/prism_training.py --CLIP_model RN101 --task retrieval --setting stereotype_fraud_race --epochs 1 --num_bases 0 --num_samples 200 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN101 --task retrieval --setting stereotype_fraud_race --epochs 1 --num_bases 0 --num_samples 200 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN101 --task retrieval --setting stereotype_fraud_race --epochs 1 --num_bases 0 --num_samples 200 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_fraud_utkrace
python debiasing_methods/prism_training.py --CLIP_model RN101 --task retrieval --setting stereotype_fraud_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN101 --task retrieval --setting stereotype_fraud_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN101 --task retrieval --setting stereotype_fraud_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_freak_gender
python debiasing_methods/prism_training.py --CLIP_model RN101 --task retrieval --setting stereotype_freak_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN101 --task retrieval --setting stereotype_freak_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN101 --task retrieval --setting stereotype_freak_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_freak_race
python debiasing_methods/prism_training.py --CLIP_model RN101 --task retrieval --setting stereotype_freak_race --epochs 1 --num_bases 0 --num_samples 200 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN101 --task retrieval --setting stereotype_freak_race --epochs 1 --num_bases 0 --num_samples 200 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN101 --task retrieval --setting stereotype_freak_race --epochs 1 --num_bases 0 --num_samples 200 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_freak_utkrace
python debiasing_methods/prism_training.py --CLIP_model RN101 --task retrieval --setting stereotype_freak_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN101 --task retrieval --setting stereotype_freak_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN101 --task retrieval --setting stereotype_freak_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_gangster_gender
python debiasing_methods/prism_training.py --CLIP_model RN101 --task retrieval --setting stereotype_gangster_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN101 --task retrieval --setting stereotype_gangster_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN101 --task retrieval --setting stereotype_gangster_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_gangster_race
python debiasing_methods/prism_training.py --CLIP_model RN101 --task retrieval --setting stereotype_gangster_race --epochs 1 --num_bases 0 --num_samples 200 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN101 --task retrieval --setting stereotype_gangster_race --epochs 1 --num_bases 0 --num_samples 200 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN101 --task retrieval --setting stereotype_gangster_race --epochs 1 --num_bases 0 --num_samples 200 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_gangster_utkrace
python debiasing_methods/prism_training.py --CLIP_model RN101 --task retrieval --setting stereotype_gangster_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN101 --task retrieval --setting stereotype_gangster_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN101 --task retrieval --setting stereotype_gangster_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_hacker_gender
python debiasing_methods/prism_training.py --CLIP_model RN101 --task retrieval --setting stereotype_hacker_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN101 --task retrieval --setting stereotype_hacker_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN101 --task retrieval --setting stereotype_hacker_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_hacker_race
python debiasing_methods/prism_training.py --CLIP_model RN101 --task retrieval --setting stereotype_hacker_race --epochs 1 --num_bases 0 --num_samples 200 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN101 --task retrieval --setting stereotype_hacker_race --epochs 1 --num_bases 0 --num_samples 200 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN101 --task retrieval --setting stereotype_hacker_race --epochs 1 --num_bases 0 --num_samples 200 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_hacker_utkrace
python debiasing_methods/prism_training.py --CLIP_model RN101 --task retrieval --setting stereotype_hacker_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN101 --task retrieval --setting stereotype_hacker_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN101 --task retrieval --setting stereotype_hacker_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_illegal-person_gender
python debiasing_methods/prism_training.py --CLIP_model RN101 --task retrieval --setting stereotype_illegal-person_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN101 --task retrieval --setting stereotype_illegal-person_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN101 --task retrieval --setting stereotype_illegal-person_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_illegal-person_race
python debiasing_methods/prism_training.py --CLIP_model RN101 --task retrieval --setting stereotype_illegal-person_race --epochs 1 --num_bases 0 --num_samples 200 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN101 --task retrieval --setting stereotype_illegal-person_race --epochs 1 --num_bases 0 --num_samples 200 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN101 --task retrieval --setting stereotype_illegal-person_race --epochs 1 --num_bases 0 --num_samples 200 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_illegal-person_utkrace
python debiasing_methods/prism_training.py --CLIP_model RN101 --task retrieval --setting stereotype_illegal-person_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN101 --task retrieval --setting stereotype_illegal-person_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN101 --task retrieval --setting stereotype_illegal-person_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_insane-person_gender
python debiasing_methods/prism_training.py --CLIP_model RN101 --task retrieval --setting stereotype_insane-person_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN101 --task retrieval --setting stereotype_insane-person_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN101 --task retrieval --setting stereotype_insane-person_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_insane-person_race
python debiasing_methods/prism_training.py --CLIP_model RN101 --task retrieval --setting stereotype_insane-person_race --epochs 1 --num_bases 0 --num_samples 200 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN101 --task retrieval --setting stereotype_insane-person_race --epochs 1 --num_bases 0 --num_samples 200 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN101 --task retrieval --setting stereotype_insane-person_race --epochs 1 --num_bases 0 --num_samples 200 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_insane-person_utkrace
python debiasing_methods/prism_training.py --CLIP_model RN101 --task retrieval --setting stereotype_insane-person_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN101 --task retrieval --setting stereotype_insane-person_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN101 --task retrieval --setting stereotype_insane-person_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_lawless-person_gender
python debiasing_methods/prism_training.py --CLIP_model RN101 --task retrieval --setting stereotype_lawless-person_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN101 --task retrieval --setting stereotype_lawless-person_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN101 --task retrieval --setting stereotype_lawless-person_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_lawless-person_race
python debiasing_methods/prism_training.py --CLIP_model RN101 --task retrieval --setting stereotype_lawless-person_race --epochs 1 --num_bases 0 --num_samples 200 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN101 --task retrieval --setting stereotype_lawless-person_race --epochs 1 --num_bases 0 --num_samples 200 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN101 --task retrieval --setting stereotype_lawless-person_race --epochs 1 --num_bases 0 --num_samples 200 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_lawless-person_utkrace
python debiasing_methods/prism_training.py --CLIP_model RN101 --task retrieval --setting stereotype_lawless-person_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN101 --task retrieval --setting stereotype_lawless-person_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN101 --task retrieval --setting stereotype_lawless-person_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_mugger_gender
python debiasing_methods/prism_training.py --CLIP_model RN101 --task retrieval --setting stereotype_mugger_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN101 --task retrieval --setting stereotype_mugger_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN101 --task retrieval --setting stereotype_mugger_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_mugger_race
python debiasing_methods/prism_training.py --CLIP_model RN101 --task retrieval --setting stereotype_mugger_race --epochs 1 --num_bases 0 --num_samples 200 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN101 --task retrieval --setting stereotype_mugger_race --epochs 1 --num_bases 0 --num_samples 200 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN101 --task retrieval --setting stereotype_mugger_race --epochs 1 --num_bases 0 --num_samples 200 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_mugger_utkrace
python debiasing_methods/prism_training.py --CLIP_model RN101 --task retrieval --setting stereotype_mugger_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN101 --task retrieval --setting stereotype_mugger_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN101 --task retrieval --setting stereotype_mugger_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_murderer_gender
python debiasing_methods/prism_training.py --CLIP_model RN101 --task retrieval --setting stereotype_murderer_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN101 --task retrieval --setting stereotype_murderer_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN101 --task retrieval --setting stereotype_murderer_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_murderer_race
python debiasing_methods/prism_training.py --CLIP_model RN101 --task retrieval --setting stereotype_murderer_race --epochs 1 --num_bases 0 --num_samples 200 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN101 --task retrieval --setting stereotype_murderer_race --epochs 1 --num_bases 0 --num_samples 200 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN101 --task retrieval --setting stereotype_murderer_race --epochs 1 --num_bases 0 --num_samples 200 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_murderer_utkrace
python debiasing_methods/prism_training.py --CLIP_model RN101 --task retrieval --setting stereotype_murderer_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN101 --task retrieval --setting stereotype_murderer_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN101 --task retrieval --setting stereotype_murderer_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_psychopath_gender
python debiasing_methods/prism_training.py --CLIP_model RN101 --task retrieval --setting stereotype_psychopath_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN101 --task retrieval --setting stereotype_psychopath_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN101 --task retrieval --setting stereotype_psychopath_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_psychopath_race
python debiasing_methods/prism_training.py --CLIP_model RN101 --task retrieval --setting stereotype_psychopath_race --epochs 1 --num_bases 0 --num_samples 200 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN101 --task retrieval --setting stereotype_psychopath_race --epochs 1 --num_bases 0 --num_samples 200 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN101 --task retrieval --setting stereotype_psychopath_race --epochs 1 --num_bases 0 --num_samples 200 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_psychopath_utkrace
python debiasing_methods/prism_training.py --CLIP_model RN101 --task retrieval --setting stereotype_psychopath_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN101 --task retrieval --setting stereotype_psychopath_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN101 --task retrieval --setting stereotype_psychopath_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_robber_gender
python debiasing_methods/prism_training.py --CLIP_model RN101 --task retrieval --setting stereotype_robber_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN101 --task retrieval --setting stereotype_robber_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN101 --task retrieval --setting stereotype_robber_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_robber_race
python debiasing_methods/prism_training.py --CLIP_model RN101 --task retrieval --setting stereotype_robber_race --epochs 1 --num_bases 0 --num_samples 200 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN101 --task retrieval --setting stereotype_robber_race --epochs 1 --num_bases 0 --num_samples 200 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN101 --task retrieval --setting stereotype_robber_race --epochs 1 --num_bases 0 --num_samples 200 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_robber_utkrace
python debiasing_methods/prism_training.py --CLIP_model RN101 --task retrieval --setting stereotype_robber_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN101 --task retrieval --setting stereotype_robber_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN101 --task retrieval --setting stereotype_robber_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_savage_gender
python debiasing_methods/prism_training.py --CLIP_model RN101 --task retrieval --setting stereotype_savage_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN101 --task retrieval --setting stereotype_savage_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN101 --task retrieval --setting stereotype_savage_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_savage_race
python debiasing_methods/prism_training.py --CLIP_model RN101 --task retrieval --setting stereotype_savage_race --epochs 1 --num_bases 0 --num_samples 200 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN101 --task retrieval --setting stereotype_savage_race --epochs 1 --num_bases 0 --num_samples 200 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN101 --task retrieval --setting stereotype_savage_race --epochs 1 --num_bases 0 --num_samples 200 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_savage_utkrace
python debiasing_methods/prism_training.py --CLIP_model RN101 --task retrieval --setting stereotype_savage_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN101 --task retrieval --setting stereotype_savage_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN101 --task retrieval --setting stereotype_savage_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_shoplifter_gender
python debiasing_methods/prism_training.py --CLIP_model RN101 --task retrieval --setting stereotype_shoplifter_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN101 --task retrieval --setting stereotype_shoplifter_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN101 --task retrieval --setting stereotype_shoplifter_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_shoplifter_race
python debiasing_methods/prism_training.py --CLIP_model RN101 --task retrieval --setting stereotype_shoplifter_race --epochs 1 --num_bases 0 --num_samples 200 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN101 --task retrieval --setting stereotype_shoplifter_race --epochs 1 --num_bases 0 --num_samples 200 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN101 --task retrieval --setting stereotype_shoplifter_race --epochs 1 --num_bases 0 --num_samples 200 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_shoplifter_utkrace
python debiasing_methods/prism_training.py --CLIP_model RN101 --task retrieval --setting stereotype_shoplifter_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN101 --task retrieval --setting stereotype_shoplifter_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN101 --task retrieval --setting stereotype_shoplifter_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_terrorist_gender
python debiasing_methods/prism_training.py --CLIP_model RN101 --task retrieval --setting stereotype_terrorist_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN101 --task retrieval --setting stereotype_terrorist_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN101 --task retrieval --setting stereotype_terrorist_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_terrorist_race
python debiasing_methods/prism_training.py --CLIP_model RN101 --task retrieval --setting stereotype_terrorist_race --epochs 1 --num_bases 0 --num_samples 200 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN101 --task retrieval --setting stereotype_terrorist_race --epochs 1 --num_bases 0 --num_samples 200 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN101 --task retrieval --setting stereotype_terrorist_race --epochs 1 --num_bases 0 --num_samples 200 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_terrorist_utkrace
python debiasing_methods/prism_training.py --CLIP_model RN101 --task retrieval --setting stereotype_terrorist_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN101 --task retrieval --setting stereotype_terrorist_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN101 --task retrieval --setting stereotype_terrorist_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_thief_gender
python debiasing_methods/prism_training.py --CLIP_model RN101 --task retrieval --setting stereotype_thief_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN101 --task retrieval --setting stereotype_thief_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN101 --task retrieval --setting stereotype_thief_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_thief_race
python debiasing_methods/prism_training.py --CLIP_model RN101 --task retrieval --setting stereotype_thief_race --epochs 1 --num_bases 0 --num_samples 200 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN101 --task retrieval --setting stereotype_thief_race --epochs 1 --num_bases 0 --num_samples 200 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN101 --task retrieval --setting stereotype_thief_race --epochs 1 --num_bases 0 --num_samples 200 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_thief_utkrace
python debiasing_methods/prism_training.py --CLIP_model RN101 --task retrieval --setting stereotype_thief_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN101 --task retrieval --setting stereotype_thief_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN101 --task retrieval --setting stereotype_thief_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_thug_gender
python debiasing_methods/prism_training.py --CLIP_model RN101 --task retrieval --setting stereotype_thug_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN101 --task retrieval --setting stereotype_thug_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN101 --task retrieval --setting stereotype_thug_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_thug_race
python debiasing_methods/prism_training.py --CLIP_model RN101 --task retrieval --setting stereotype_thug_race --epochs 1 --num_bases 0 --num_samples 200 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN101 --task retrieval --setting stereotype_thug_race --epochs 1 --num_bases 0 --num_samples 200 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN101 --task retrieval --setting stereotype_thug_race --epochs 1 --num_bases 0 --num_samples 200 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_thug_utkrace
python debiasing_methods/prism_training.py --CLIP_model RN101 --task retrieval --setting stereotype_thug_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN101 --task retrieval --setting stereotype_thug_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN101 --task retrieval --setting stereotype_thug_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_violent-person_gender
python debiasing_methods/prism_training.py --CLIP_model RN101 --task retrieval --setting stereotype_violent-person_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN101 --task retrieval --setting stereotype_violent-person_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN101 --task retrieval --setting stereotype_violent-person_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_violent-person_race
python debiasing_methods/prism_training.py --CLIP_model RN101 --task retrieval --setting stereotype_violent-person_race --epochs 1 --num_bases 0 --num_samples 200 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN101 --task retrieval --setting stereotype_violent-person_race --epochs 1 --num_bases 0 --num_samples 200 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN101 --task retrieval --setting stereotype_violent-person_race --epochs 1 --num_bases 0 --num_samples 200 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: stereotype_violent-person_utkrace
python debiasing_methods/prism_training.py --CLIP_model RN101 --task retrieval --setting stereotype_violent-person_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN101 --task retrieval --setting stereotype_violent-person_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN101 --task retrieval --setting stereotype_violent-person_utkrace --epochs 1 --num_bases 0 --num_samples 200 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: celeba_black_hair_gender
python debiasing_methods/prism_training.py --CLIP_model RN101 --task retrieval --setting celeba_black_hair_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN101 --task retrieval --setting celeba_black_hair_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN101 --task retrieval --setting celeba_black_hair_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: celeba_blond_hair_gender
python debiasing_methods/prism_training.py --CLIP_model RN101 --task retrieval --setting celeba_blond_hair_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN101 --task retrieval --setting celeba_blond_hair_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN101 --task retrieval --setting celeba_blond_hair_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: celeba_brown_hair_gender
python debiasing_methods/prism_training.py --CLIP_model RN101 --task retrieval --setting celeba_brown_hair_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN101 --task retrieval --setting celeba_brown_hair_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN101 --task retrieval --setting celeba_brown_hair_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: celeba_gray_hair_gender
python debiasing_methods/prism_training.py --CLIP_model RN101 --task retrieval --setting celeba_gray_hair_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN101 --task retrieval --setting celeba_gray_hair_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN101 --task retrieval --setting celeba_gray_hair_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random



# Zero-shot classification tasks
## Backbone: RN50
### Setting: celeba_blond_hair_gender
python debiasing_methods/prism_training.py --CLIP_model RN50 --task zs_classification --setting celeba_blond_hair_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN50 --task zs_classification --setting celeba_blond_hair_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN50 --task zs_classification --setting celeba_blond_hair_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: waterbirds
python debiasing_methods/prism_training.py --CLIP_model RN50 --task zs_classification --setting waterbirds --epochs 1 --num_bases 0 --num_samples 500 --seed 0 --lr 0.1 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN50 --task zs_classification --setting waterbirds --epochs 1 --num_bases 0 --num_samples 500 --seed 1 --lr 0.1 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN50 --task zs_classification --setting waterbirds --epochs 1 --num_bases 0 --num_samples 500 --seed 2 --lr 0.1 --batch_size 64 --wd 0 --init_weight random


## Backbone: RN101
### Setting: celeba_blond_hair_gender
python debiasing_methods/prism_training.py --CLIP_model RN101 --task zs_classification --setting celeba_blond_hair_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 0 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN101 --task zs_classification --setting celeba_blond_hair_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 1 --lr 0.01 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN101 --task zs_classification --setting celeba_blond_hair_gender --epochs 1 --num_bases 0 --num_samples 100 --seed 2 --lr 0.01 --batch_size 64 --wd 0 --init_weight random

### Setting: waterbirds
python debiasing_methods/prism_training.py --CLIP_model RN101 --task zs_classification --setting waterbirds --epochs 1 --num_bases 0 --num_samples 500 --seed 0 --lr 0.1 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN101 --task zs_classification --setting waterbirds --epochs 1 --num_bases 0 --num_samples 500 --seed 1 --lr 0.1 --batch_size 64 --wd 0 --init_weight random
python debiasing_methods/prism_training.py --CLIP_model RN101 --task zs_classification --setting waterbirds --epochs 1 --num_bases 0 --num_samples 500 --seed 2 --lr 0.1 --batch_size 64 --wd 0 --init_weight random
