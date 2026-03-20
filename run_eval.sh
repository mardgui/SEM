# Retrieval results (Table 2)
## FairFace
### ViT-B/16
#### Race bias
##### Base CLIP
python main_eval.py --config config/fairface_retrieval_stereotype_race_clip-ViT-B-16_vanilla.yml

##### Bias-agnostic + input-specific prompts
python main_eval.py --config config/fairface_retrieval_stereotype_race_clip-ViT-B-16_roboshot.yml
python main_eval.py --config config/fairface_retrieval_stereotype_race_clip-ViT-B-16_sem-i.yml

##### Bias prompts only
python main_eval.py --config config/fairface_retrieval_stereotype_race_clip-ViT-B-16_orth-proj.yml
python main_eval.py --config config/fairface_retrieval_stereotype_race_clip-ViT-B-16_prism-mini.yml
python main_eval.py --config config/fairface_retrieval_stereotype_race_clip-ViT-B-16_sem-b.yml
python main_eval.py --config config/fairface_retrieval_stereotype_race_clip-ViT-B-16_zsdebias-seed0.yml
python main_eval.py --config config/fairface_retrieval_stereotype_race_clip-ViT-B-16_zsdebias-seed1.yml
python main_eval.py --config config/fairface_retrieval_stereotype_race_clip-ViT-B-16_zsdebias-seed2.yml

##### Bias prompts + input-specific prompts (average over 3 seeds for PRISM)
python main_eval.py --config config/fairface_retrieval_stereotype_race_clip-ViT-B-16_orth-cali.yml
python main_eval.py --config config/fairface_retrieval_stereotype_race_clip-ViT-B-16_sem-bi.yml
python main_eval.py --config config/fairface_retrieval_stereotype_race_clip-ViT-B-16_prism-seed0.yml
python main_eval.py --config config/fairface_retrieval_stereotype_race_clip-ViT-B-16_prism-seed1.yml
python main_eval.py --config config/fairface_retrieval_stereotype_race_clip-ViT-B-16_prism-seed2.yml

##### Bias prompts + input-specific prompts + labeled images
python main_eval.py --config config/fairface_retrieval_stereotype_race_clip-ViT-B-16_bendvlm.yml
python main_eval.py --config config/fairface_retrieval_stereotype_race_clip-ViT-B-16_bendsem-bi.yml


#### Gender bias
##### Base CLIP
python main_eval.py --config config/fairface_retrieval_stereotype_gender_clip-ViT-B-16_vanilla.yml

##### Bias-agnostic + input-specific prompts
python main_eval.py --config config/fairface_retrieval_stereotype_gender_clip-ViT-B-16_roboshot.yml
python main_eval.py --config config/fairface_retrieval_stereotype_gender_clip-ViT-B-16_sem-i.yml

##### Bias prompts only
python main_eval.py --config config/fairface_retrieval_stereotype_gender_clip-ViT-B-16_orth-proj.yml
python main_eval.py --config config/fairface_retrieval_stereotype_gender_clip-ViT-B-16_prism-mini.yml
python main_eval.py --config config/fairface_retrieval_stereotype_gender_clip-ViT-B-16_sem-b.yml
python main_eval.py --config config/fairface_retrieval_stereotype_gender_clip-ViT-B-16_zsdebias-seed0.yml
python main_eval.py --config config/fairface_retrieval_stereotype_gender_clip-ViT-B-16_zsdebias-seed1.yml
python main_eval.py --config config/fairface_retrieval_stereotype_gender_clip-ViT-B-16_zsdebias-seed2.yml

##### Bias prompts + input-specific prompts (average over 3 seeds for PRISM)
python main_eval.py --config config/fairface_retrieval_stereotype_gender_clip-ViT-B-16_orth-cali.yml
python main_eval.py --config config/fairface_retrieval_stereotype_gender_clip-ViT-B-16_sem-bi.yml
python main_eval.py --config config/fairface_retrieval_stereotype_gender_clip-ViT-B-16_prism-seed0.yml
python main_eval.py --config config/fairface_retrieval_stereotype_gender_clip-ViT-B-16_prism-seed1.yml
python main_eval.py --config config/fairface_retrieval_stereotype_gender_clip-ViT-B-16_prism-seed2.yml

##### Bias prompts + input-specific prompts + labeled images
python main_eval.py --config config/fairface_retrieval_stereotype_gender_clip-ViT-B-16_bendvlm.yml
python main_eval.py --config config/fairface_retrieval_stereotype_gender_clip-ViT-B-16_bendsem-bi.yml



### ViT-L/14@336px
#### Race bias
##### Base CLIP
python main_eval.py --config config/fairface_retrieval_stereotype_race_clip-ViT-L-14_vanilla.yml

##### Bias-agnostic + input-specific prompts
python main_eval.py --config config/fairface_retrieval_stereotype_race_clip-ViT-L-14_roboshot.yml
python main_eval.py --config config/fairface_retrieval_stereotype_race_clip-ViT-L-14_sem-i.yml

##### Bias prompts only
python main_eval.py --config config/fairface_retrieval_stereotype_race_clip-ViT-L-14_orth-proj.yml
python main_eval.py --config config/fairface_retrieval_stereotype_race_clip-ViT-L-14_prism-mini.yml
python main_eval.py --config config/fairface_retrieval_stereotype_race_clip-ViT-L-14_sem-b.yml
python main_eval.py --config config/fairface_retrieval_stereotype_race_clip-ViT-L-14_zsdebias-seed0.yml
python main_eval.py --config config/fairface_retrieval_stereotype_race_clip-ViT-L-14_zsdebias-seed1.yml
python main_eval.py --config config/fairface_retrieval_stereotype_race_clip-ViT-L-14_zsdebias-seed2.yml

##### Bias prompts + input-specific prompts (average over 3 seeds for PRISM)
python main_eval.py --config config/fairface_retrieval_stereotype_race_clip-ViT-L-14_orth-cali.yml
python main_eval.py --config config/fairface_retrieval_stereotype_race_clip-ViT-L-14_sem-bi.yml
python main_eval.py --config config/fairface_retrieval_stereotype_race_clip-ViT-L-14_prism-seed0.yml
python main_eval.py --config config/fairface_retrieval_stereotype_race_clip-ViT-L-14_prism-seed1.yml
python main_eval.py --config config/fairface_retrieval_stereotype_race_clip-ViT-L-14_prism-seed2.yml

##### Bias prompts + input-specific prompts + labeled images
python main_eval.py --config config/fairface_retrieval_stereotype_race_clip-ViT-L-14_bendvlm.yml
python main_eval.py --config config/fairface_retrieval_stereotype_race_clip-ViT-L-14_bendsem-bi.yml


#### Gender bias
##### Base CLIP
python main_eval.py --config config/fairface_retrieval_stereotype_gender_clip-ViT-L-14_vanilla.yml

##### Bias-agnostic + input-specific prompts
python main_eval.py --config config/fairface_retrieval_stereotype_gender_clip-ViT-L-14_roboshot.yml
python main_eval.py --config config/fairface_retrieval_stereotype_gender_clip-ViT-L-14_sem-i.yml

##### Bias prompts only
python main_eval.py --config config/fairface_retrieval_stereotype_gender_clip-ViT-L-14_orth-proj.yml
python main_eval.py --config config/fairface_retrieval_stereotype_gender_clip-ViT-L-14_prism-mini.yml
python main_eval.py --config config/fairface_retrieval_stereotype_gender_clip-ViT-L-14_sem-b.yml
python main_eval.py --config config/fairface_retrieval_stereotype_gender_clip-ViT-L-14_zsdebias-seed0.yml
python main_eval.py --config config/fairface_retrieval_stereotype_gender_clip-ViT-L-14_zsdebias-seed1.yml
python main_eval.py --config config/fairface_retrieval_stereotype_gender_clip-ViT-L-14_zsdebias-seed2.yml

##### Bias prompts + input-specific prompts (average over 3 seeds for PRISM)
python main_eval.py --config config/fairface_retrieval_stereotype_gender_clip-ViT-L-14_orth-cali.yml
python main_eval.py --config config/fairface_retrieval_stereotype_gender_clip-ViT-L-14_sem-bi.yml
python main_eval.py --config config/fairface_retrieval_stereotype_gender_clip-ViT-L-14_prism-seed0.yml
python main_eval.py --config config/fairface_retrieval_stereotype_gender_clip-ViT-L-14_prism-seed1.yml
python main_eval.py --config config/fairface_retrieval_stereotype_gender_clip-ViT-L-14_prism-seed2.yml

##### Bias prompts + input-specific prompts + labeled images
python main_eval.py --config config/fairface_retrieval_stereotype_gender_clip-ViT-L-14_bendvlm.yml
python main_eval.py --config config/fairface_retrieval_stereotype_gender_clip-ViT-L-14_bendsem-bi.yml




## UTKFace
### ViT-B/16
#### Race bias
##### Base CLIP
python main_eval.py --config config/utkface_retrieval_stereotype_race_clip-ViT-B-16_vanilla.yml

##### Bias-agnostic + input-specific prompts
python main_eval.py --config config/utkface_retrieval_stereotype_race_clip-ViT-B-16_roboshot.yml
python main_eval.py --config config/utkface_retrieval_stereotype_race_clip-ViT-B-16_sem-i.yml

##### Bias prompts only
python main_eval.py --config config/utkface_retrieval_stereotype_race_clip-ViT-B-16_orth-proj.yml
python main_eval.py --config config/utkface_retrieval_stereotype_race_clip-ViT-B-16_prism-mini.yml
python main_eval.py --config config/utkface_retrieval_stereotype_race_clip-ViT-B-16_sem-b.yml
python main_eval.py --config config/utkface_retrieval_stereotype_race_clip-ViT-B-16_zsdebias-seed0.yml
python main_eval.py --config config/utkface_retrieval_stereotype_race_clip-ViT-B-16_zsdebias-seed1.yml
python main_eval.py --config config/utkface_retrieval_stereotype_race_clip-ViT-B-16_zsdebias-seed2.yml

##### Bias prompts + input-specific prompts (average over 3 seeds for PRISM)
python main_eval.py --config config/utkface_retrieval_stereotype_race_clip-ViT-B-16_orth-cali.yml
python main_eval.py --config config/utkface_retrieval_stereotype_race_clip-ViT-B-16_sem-bi.yml
python main_eval.py --config config/utkface_retrieval_stereotype_race_clip-ViT-B-16_prism-seed0.yml
python main_eval.py --config config/utkface_retrieval_stereotype_race_clip-ViT-B-16_prism-seed1.yml
python main_eval.py --config config/utkface_retrieval_stereotype_race_clip-ViT-B-16_prism-seed2.yml

##### Bias prompts + input-specific prompts + labeled images
python main_eval.py --config config/utkface_retrieval_stereotype_race_clip-ViT-B-16_bendvlm.yml
python main_eval.py --config config/utkface_retrieval_stereotype_race_clip-ViT-B-16_bendsem-bi.yml


#### Gender bias
##### Base CLIP
python main_eval.py --config config/utkface_retrieval_stereotype_gender_clip-ViT-B-16_vanilla.yml

##### Bias-agnostic + input-specific prompts
python main_eval.py --config config/utkface_retrieval_stereotype_gender_clip-ViT-B-16_roboshot.yml
python main_eval.py --config config/utkface_retrieval_stereotype_gender_clip-ViT-B-16_sem-i.yml

##### Bias prompts only
python main_eval.py --config config/utkface_retrieval_stereotype_gender_clip-ViT-B-16_orth-proj.yml
python main_eval.py --config config/utkface_retrieval_stereotype_gender_clip-ViT-B-16_prism-mini.yml
python main_eval.py --config config/utkface_retrieval_stereotype_gender_clip-ViT-B-16_sem-b.yml
python main_eval.py --config config/utkface_retrieval_stereotype_gender_clip-ViT-B-16_zsdebias-seed0.yml
python main_eval.py --config config/utkface_retrieval_stereotype_gender_clip-ViT-B-16_zsdebias-seed1.yml
python main_eval.py --config config/utkface_retrieval_stereotype_gender_clip-ViT-B-16_zsdebias-seed2.yml

##### Bias prompts + input-specific prompts (average over 3 seeds for PRISM)
python main_eval.py --config config/utkface_retrieval_stereotype_gender_clip-ViT-B-16_orth-cali.yml
python main_eval.py --config config/utkface_retrieval_stereotype_gender_clip-ViT-B-16_sem-bi.yml
python main_eval.py --config config/utkface_retrieval_stereotype_gender_clip-ViT-B-16_prism-seed0.yml
python main_eval.py --config config/utkface_retrieval_stereotype_gender_clip-ViT-B-16_prism-seed1.yml
python main_eval.py --config config/utkface_retrieval_stereotype_gender_clip-ViT-B-16_prism-seed2.yml

##### Bias prompts + input-specific prompts + labeled images
python main_eval.py --config config/utkface_retrieval_stereotype_gender_clip-ViT-B-16_bendvlm.yml
python main_eval.py --config config/utkface_retrieval_stereotype_gender_clip-ViT-B-16_bendsem-bi.yml



### ViT-L/14@336px
#### Race bias
##### Base CLIP
python main_eval.py --config config/utkface_retrieval_stereotype_race_clip-ViT-L-14_vanilla.yml

##### Bias-agnostic + input-specific prompts
python main_eval.py --config config/utkface_retrieval_stereotype_race_clip-ViT-L-14_roboshot.yml
python main_eval.py --config config/utkface_retrieval_stereotype_race_clip-ViT-L-14_sem-i.yml

##### Bias prompts only
python main_eval.py --config config/utkface_retrieval_stereotype_race_clip-ViT-L-14_orth-proj.yml
python main_eval.py --config config/utkface_retrieval_stereotype_race_clip-ViT-L-14_prism-mini.yml
python main_eval.py --config config/utkface_retrieval_stereotype_race_clip-ViT-L-14_sem-b.yml
python main_eval.py --config config/utkface_retrieval_stereotype_race_clip-ViT-L-14_zsdebias-seed0.yml
python main_eval.py --config config/utkface_retrieval_stereotype_race_clip-ViT-L-14_zsdebias-seed1.yml
python main_eval.py --config config/utkface_retrieval_stereotype_race_clip-ViT-L-14_zsdebias-seed2.yml

##### Bias prompts + input-specific prompts (average over 3 seeds for PRISM)
python main_eval.py --config config/utkface_retrieval_stereotype_race_clip-ViT-L-14_orth-cali.yml
python main_eval.py --config config/utkface_retrieval_stereotype_race_clip-ViT-L-14_sem-bi.yml
python main_eval.py --config config/utkface_retrieval_stereotype_race_clip-ViT-L-14_prism-seed0.yml
python main_eval.py --config config/utkface_retrieval_stereotype_race_clip-ViT-L-14_prism-seed1.yml
python main_eval.py --config config/utkface_retrieval_stereotype_race_clip-ViT-L-14_prism-seed2.yml

##### Bias prompts + input-specific prompts + labeled images
python main_eval.py --config config/utkface_retrieval_stereotype_race_clip-ViT-L-14_bendvlm.yml
python main_eval.py --config config/utkface_retrieval_stereotype_race_clip-ViT-L-14_bendsem-bi.yml


#### Gender bias
##### Base CLIP
python main_eval.py --config config/utkface_retrieval_stereotype_gender_clip-ViT-L-14_vanilla.yml

##### Bias-agnostic + input-specific prompts
python main_eval.py --config config/utkface_retrieval_stereotype_gender_clip-ViT-L-14_roboshot.yml
python main_eval.py --config config/utkface_retrieval_stereotype_gender_clip-ViT-L-14_sem-i.yml

##### Bias prompts only
python main_eval.py --config config/utkface_retrieval_stereotype_gender_clip-ViT-L-14_orth-proj.yml
python main_eval.py --config config/utkface_retrieval_stereotype_gender_clip-ViT-L-14_prism-mini.yml
python main_eval.py --config config/utkface_retrieval_stereotype_gender_clip-ViT-L-14_sem-b.yml
python main_eval.py --config config/utkface_retrieval_stereotype_gender_clip-ViT-L-14_zsdebias-seed0.yml
python main_eval.py --config config/utkface_retrieval_stereotype_gender_clip-ViT-L-14_zsdebias-seed1.yml
python main_eval.py --config config/utkface_retrieval_stereotype_gender_clip-ViT-L-14_zsdebias-seed2.yml

##### Bias prompts + input-specific prompts (average over 3 seeds for PRISM)
python main_eval.py --config config/utkface_retrieval_stereotype_gender_clip-ViT-L-14_orth-cali.yml
python main_eval.py --config config/utkface_retrieval_stereotype_gender_clip-ViT-L-14_sem-bi.yml
python main_eval.py --config config/utkface_retrieval_stereotype_gender_clip-ViT-L-14_prism-seed0.yml
python main_eval.py --config config/utkface_retrieval_stereotype_gender_clip-ViT-L-14_prism-seed1.yml
python main_eval.py --config config/utkface_retrieval_stereotype_gender_clip-ViT-L-14_prism-seed2.yml

##### Bias prompts + input-specific prompts + labeled images
python main_eval.py --config config/utkface_retrieval_stereotype_gender_clip-ViT-L-14_bendvlm.yml
python main_eval.py --config config/utkface_retrieval_stereotype_gender_clip-ViT-L-14_bendsem-bi.yml





# Zero-shot classification results (Table 3)
## CelebA (gender bias)
### ViT-B/16
#### Base CLIP
python main_eval.py --config config/celeba_zs-classification_celeba-blond-hair-gender_clip-ViT-B-16_vanilla.yml

#### Bias-agnostic + input-specific prompts
python main_eval.py --config config/celeba_zs-classification_celeba-blond-hair-gender_clip-ViT-B-16_roboshot.yml
python main_eval.py --config config/celeba_zs-classification_celeba-blond-hair-gender_clip-ViT-B-16_sem-i.yml

#### Bias prompts only
python main_eval.py --config config/celeba_zs-classification_celeba-blond-hair-gender_clip-ViT-B-16_orth-proj.yml
python main_eval.py --config config/celeba_zs-classification_celeba-blond-hair-gender_clip-ViT-B-16_prism-mini.yml
python main_eval.py --config config/celeba_zs-classification_celeba-blond-hair-gender_clip-ViT-B-16_sem-b.yml
python main_eval.py --config config/celeba_zs-classification_celeba-blond-hair-gender_clip-ViT-B-16_zsdebias-seed0.yml
python main_eval.py --config config/celeba_zs-classification_celeba-blond-hair-gender_clip-ViT-B-16_zsdebias-seed1.yml
python main_eval.py --config config/celeba_zs-classification_celeba-blond-hair-gender_clip-ViT-B-16_zsdebias-seed2.yml

#### Bias prompts + input-specific prompts (average over 3 seeds for PRISM)
python main_eval.py --config config/celeba_zs-classification_celeba-blond-hair-gender_clip-ViT-B-16_orth-cali.yml
python main_eval.py --config config/celeba_zs-classification_celeba-blond-hair-gender_clip-ViT-B-16_sem-bi.yml
python main_eval.py --config config/celeba_zs-classification_celeba-blond-hair-gender_clip-ViT-B-16_prism-seed0.yml
python main_eval.py --config config/celeba_zs-classification_celeba-blond-hair-gender_clip-ViT-B-16_prism-seed1.yml
python main_eval.py --config config/celeba_zs-classification_celeba-blond-hair-gender_clip-ViT-B-16_prism-seed2.yml

#### Bias prompts + input-specific prompts + labeled images
python main_eval.py --config config/celeba_zs-classification_celeba-blond-hair-gender_clip-ViT-B-16_bendvlm.yml
python main_eval.py --config config/celeba_zs-classification_celeba-blond-hair-gender_clip-ViT-B-16_bendsem-bi.yml


### ViT-L/14@336px
#### Base CLIP
python main_eval.py --config config/celeba_zs-classification_celeba-blond-hair-gender_clip-ViT-L-14_vanilla.yml

#### Bias-agnostic + input-specific prompts
python main_eval.py --config config/celeba_zs-classification_celeba-blond-hair-gender_clip-ViT-L-14_roboshot.yml
python main_eval.py --config config/celeba_zs-classification_celeba-blond-hair-gender_clip-ViT-L-14_sem-i.yml

#### Bias prompts only
python main_eval.py --config config/celeba_zs-classification_celeba-blond-hair-gender_clip-ViT-L-14_orth-proj.yml
python main_eval.py --config config/celeba_zs-classification_celeba-blond-hair-gender_clip-ViT-L-14_prism-mini.yml
python main_eval.py --config config/celeba_zs-classification_celeba-blond-hair-gender_clip-ViT-L-14_sem-b.yml
python main_eval.py --config config/celeba_zs-classification_celeba-blond-hair-gender_clip-ViT-L-14_zsdebias-seed0.yml
python main_eval.py --config config/celeba_zs-classification_celeba-blond-hair-gender_clip-ViT-L-14_zsdebias-seed1.yml
python main_eval.py --config config/celeba_zs-classification_celeba-blond-hair-gender_clip-ViT-L-14_zsdebias-seed2.yml

#### Bias prompts + input-specific prompts (average over 3 seeds for PRISM)
python main_eval.py --config config/celeba_zs-classification_celeba-blond-hair-gender_clip-ViT-L-14_orth-cali.yml
python main_eval.py --config config/celeba_zs-classification_celeba-blond-hair-gender_clip-ViT-L-14_sem-bi.yml
python main_eval.py --config config/celeba_zs-classification_celeba-blond-hair-gender_clip-ViT-L-14_prism-seed0.yml
python main_eval.py --config config/celeba_zs-classification_celeba-blond-hair-gender_clip-ViT-L-14_prism-seed1.yml
python main_eval.py --config config/celeba_zs-classification_celeba-blond-hair-gender_clip-ViT-L-14_prism-seed2.yml

#### Bias prompts + input-specific prompts + labeled images
python main_eval.py --config config/celeba_zs-classification_celeba-blond-hair-gender_clip-ViT-L-14_bendvlm.yml
python main_eval.py --config config/celeba_zs-classification_celeba-blond-hair-gender_clip-ViT-L-14_bendsem-bi.yml



## Waterbirds (background bias)
### ViT-B/16
#### Base CLIP
python main_eval.py --config config/waterbirds_zs-classification_waterbirds_clip-ViT-B-16_vanilla.yml

#### Bias-agnostic + input-specific prompts
python main_eval.py --config config/waterbirds_zs-classification_waterbirds_clip-ViT-B-16_roboshot.yml
python main_eval.py --config config/waterbirds_zs-classification_waterbirds_clip-ViT-B-16_sem-i.yml

#### Bias prompts only
python main_eval.py --config config/waterbirds_zs-classification_waterbirds_clip-ViT-B-16_orth-proj.yml
python main_eval.py --config config/waterbirds_zs-classification_waterbirds_clip-ViT-B-16_prism-mini.yml
python main_eval.py --config config/waterbirds_zs-classification_waterbirds_clip-ViT-B-16_sem-b.yml
python main_eval.py --config config/waterbirds_zs-classification_waterbirds_clip-ViT-B-16_zsdebias-seed0.yml
python main_eval.py --config config/waterbirds_zs-classification_waterbirds_clip-ViT-B-16_zsdebias-seed1.yml
python main_eval.py --config config/waterbirds_zs-classification_waterbirds_clip-ViT-B-16_zsdebias-seed2.yml

#### Bias prompts + input-specific prompts (average over 3 seeds for PRISM)
python main_eval.py --config config/waterbirds_zs-classification_waterbirds_clip-ViT-B-16_orth-cali.yml
python main_eval.py --config config/waterbirds_zs-classification_waterbirds_clip-ViT-B-16_sem-bi.yml
python main_eval.py --config config/waterbirds_zs-classification_waterbirds_clip-ViT-B-16_prism-seed0.yml
python main_eval.py --config config/waterbirds_zs-classification_waterbirds_clip-ViT-B-16_prism-seed1.yml
python main_eval.py --config config/waterbirds_zs-classification_waterbirds_clip-ViT-B-16_prism-seed2.yml

#### Bias prompts + input-specific prompts + labeled images
python main_eval.py --config config/waterbirds_zs-classification_waterbirds_clip-ViT-B-16_bendvlm.yml
python main_eval.py --config config/waterbirds_zs-classification_waterbirds_clip-ViT-B-16_bendsem-bi.yml


### ViT-L/14@336px
#### Base CLIP
python main_eval.py --config config/waterbirds_zs-classification_waterbirds_clip-ViT-L-14_vanilla.yml

#### Bias-agnostic + input-specific prompts
python main_eval.py --config config/waterbirds_zs-classification_waterbirds_clip-ViT-L-14_roboshot.yml
python main_eval.py --config config/waterbirds_zs-classification_waterbirds_clip-ViT-L-14_sem-i.yml

#### Bias prompts only
python main_eval.py --config config/waterbirds_zs-classification_waterbirds_clip-ViT-L-14_orth-proj.yml
python main_eval.py --config config/waterbirds_zs-classification_waterbirds_clip-ViT-L-14_prism-mini.yml
python main_eval.py --config config/waterbirds_zs-classification_waterbirds_clip-ViT-L-14_sem-b.yml
python main_eval.py --config config/waterbirds_zs-classification_waterbirds_clip-ViT-L-14_zsdebias-seed0.yml
python main_eval.py --config config/waterbirds_zs-classification_waterbirds_clip-ViT-L-14_zsdebias-seed1.yml
python main_eval.py --config config/waterbirds_zs-classification_waterbirds_clip-ViT-L-14_zsdebias-seed2.yml

#### Bias prompts + input-specific prompts (average over 3 seeds for PRISM)
python main_eval.py --config config/waterbirds_zs-classification_waterbirds_clip-ViT-L-14_orth-cali.yml
python main_eval.py --config config/waterbirds_zs-classification_waterbirds_clip-ViT-L-14_sem-bi.yml
python main_eval.py --config config/waterbirds_zs-classification_waterbirds_clip-ViT-L-14_prism-seed0.yml
python main_eval.py --config config/waterbirds_zs-classification_waterbirds_clip-ViT-L-14_prism-seed1.yml
python main_eval.py --config config/waterbirds_zs-classification_waterbirds_clip-ViT-L-14_prism-seed2.yml

#### Bias prompts + input-specific prompts + labeled images
python main_eval.py --config config/waterbirds_zs-classification_waterbirds_clip-ViT-L-14_bendvlm.yml
python main_eval.py --config config/waterbirds_zs-classification_waterbirds_clip-ViT-L-14_bendsem-bi.yml





# Additional retrieval results on CelebA (Supp. Mat.)
## ViT-B/16
### Stereotype queries
##### Base CLIP
python main_eval.py --config config/celeba_retrieval_stereotype_gender_clip-ViT-B-16_vanilla.yml

##### Bias-agnostic + input-specific prompts
python main_eval.py --config config/celeba_retrieval_stereotype_gender_clip-ViT-B-16_roboshot.yml
python main_eval.py --config config/celeba_retrieval_stereotype_gender_clip-ViT-B-16_sem-i.yml

##### Bias prompts only
python main_eval.py --config config/celeba_retrieval_stereotype_gender_clip-ViT-B-16_orth-proj.yml
python main_eval.py --config config/celeba_retrieval_stereotype_gender_clip-ViT-B-16_prism-mini.yml
python main_eval.py --config config/celeba_retrieval_stereotype_gender_clip-ViT-B-16_sem-b.yml
python main_eval.py --config config/celeba_retrieval_stereotype_gender_clip-ViT-B-16_zsdebias-seed0.yml
python main_eval.py --config config/celeba_retrieval_stereotype_gender_clip-ViT-B-16_zsdebias-seed1.yml
python main_eval.py --config config/celeba_retrieval_stereotype_gender_clip-ViT-B-16_zsdebias-seed2.yml

##### Bias prompts + input-specific prompts (average over 3 seeds for PRISM)
python main_eval.py --config config/celeba_retrieval_stereotype_gender_clip-ViT-B-16_orth-cali.yml
python main_eval.py --config config/celeba_retrieval_stereotype_gender_clip-ViT-B-16_sem-bi.yml
python main_eval.py --config config/celeba_retrieval_stereotype_gender_clip-ViT-B-16_prism-seed0.yml
python main_eval.py --config config/celeba_retrieval_stereotype_gender_clip-ViT-B-16_prism-seed1.yml
python main_eval.py --config config/celeba_retrieval_stereotype_gender_clip-ViT-B-16_prism-seed2.yml

##### Bias prompts + input-specific prompts + labeled images
python main_eval.py --config config/celeba_retrieval_stereotype_gender_clip-ViT-B-16_bendvlm.yml
python main_eval.py --config config/celeba_retrieval_stereotype_gender_clip-ViT-B-16_bendsem-bi.yml


### Hair color queries
##### Base CLIP
python main_eval.py --config config/celeba_retrieval_hair_gender_clip-ViT-B-16_vanilla.yml

##### Bias-agnostic + input-specific prompts
python main_eval.py --config config/celeba_retrieval_hair_gender_clip-ViT-B-16_roboshot.yml
python main_eval.py --config config/celeba_retrieval_hair_gender_clip-ViT-B-16_sem-i.yml

##### Bias prompts only
python main_eval.py --config config/celeba_retrieval_hair_gender_clip-ViT-B-16_orth-proj.yml
python main_eval.py --config config/celeba_retrieval_hair_gender_clip-ViT-B-16_prism-mini.yml
python main_eval.py --config config/celeba_retrieval_hair_gender_clip-ViT-B-16_sem-b.yml
python main_eval.py --config config/celeba_retrieval_hair_gender_clip-ViT-B-16_zsdebias-seed0.yml
python main_eval.py --config config/celeba_retrieval_hair_gender_clip-ViT-B-16_zsdebias-seed1.yml
python main_eval.py --config config/celeba_retrieval_hair_gender_clip-ViT-B-16_zsdebias-seed2.yml

##### Bias prompts + input-specific prompts (average over 3 seeds for PRISM)
python main_eval.py --config config/celeba_retrieval_hair_gender_clip-ViT-B-16_orth-cali.yml
python main_eval.py --config config/celeba_retrieval_hair_gender_clip-ViT-B-16_sem-bi.yml
python main_eval.py --config config/celeba_retrieval_hair_gender_clip-ViT-B-16_prism-seed0.yml
python main_eval.py --config config/celeba_retrieval_hair_gender_clip-ViT-B-16_prism-seed1.yml
python main_eval.py --config config/celeba_retrieval_hair_gender_clip-ViT-B-16_prism-seed2.yml

##### Bias prompts + input-specific prompts + labeled images
python main_eval.py --config config/celeba_retrieval_hair_gender_clip-ViT-B-16_bendvlm.yml
python main_eval.py --config config/celeba_retrieval_hair_gender_clip-ViT-B-16_bendsem-bi.yml



## ViT-L/14@336px
### Stereotype queries
##### Base CLIP
python main_eval.py --config config/celeba_retrieval_stereotype_gender_clip-ViT-L-14_vanilla.yml

##### Bias-agnostic + input-specific prompts
python main_eval.py --config config/celeba_retrieval_stereotype_gender_clip-ViT-L-14_roboshot.yml
python main_eval.py --config config/celeba_retrieval_stereotype_gender_clip-ViT-L-14_sem-i.yml

##### Bias prompts only
python main_eval.py --config config/celeba_retrieval_stereotype_gender_clip-ViT-L-14_orth-proj.yml
python main_eval.py --config config/celeba_retrieval_stereotype_gender_clip-ViT-L-14_prism-mini.yml
python main_eval.py --config config/celeba_retrieval_stereotype_gender_clip-ViT-L-14_sem-b.yml
python main_eval.py --config config/celeba_retrieval_stereotype_gender_clip-ViT-L-14_zsdebias-seed0.yml
python main_eval.py --config config/celeba_retrieval_stereotype_gender_clip-ViT-L-14_zsdebias-seed1.yml
python main_eval.py --config config/celeba_retrieval_stereotype_gender_clip-ViT-L-14_zsdebias-seed2.yml

##### Bias prompts + input-specific prompts (average over 3 seeds for PRISM)
python main_eval.py --config config/celeba_retrieval_stereotype_gender_clip-ViT-L-14_orth-cali.yml
python main_eval.py --config config/celeba_retrieval_stereotype_gender_clip-ViT-L-14_sem-bi.yml
python main_eval.py --config config/celeba_retrieval_stereotype_gender_clip-ViT-L-14_prism-seed0.yml
python main_eval.py --config config/celeba_retrieval_stereotype_gender_clip-ViT-L-14_prism-seed1.yml
python main_eval.py --config config/celeba_retrieval_stereotype_gender_clip-ViT-L-14_prism-seed2.yml

##### Bias prompts + input-specific prompts + labeled images
python main_eval.py --config config/celeba_retrieval_stereotype_gender_clip-ViT-L-14_bendvlm.yml
python main_eval.py --config config/celeba_retrieval_stereotype_gender_clip-ViT-L-14_bendsem-bi.yml


### Hair color queries
##### Base CLIP
python main_eval.py --config config/celeba_retrieval_hair_gender_clip-ViT-L-14_vanilla.yml

##### Bias-agnostic + input-specific prompts
python main_eval.py --config config/celeba_retrieval_hair_gender_clip-ViT-L-14_roboshot.yml
python main_eval.py --config config/celeba_retrieval_hair_gender_clip-ViT-L-14_sem-i.yml

##### Bias prompts only
python main_eval.py --config config/celeba_retrieval_hair_gender_clip-ViT-L-14_orth-proj.yml
python main_eval.py --config config/celeba_retrieval_hair_gender_clip-ViT-L-14_prism-mini.yml
python main_eval.py --config config/celeba_retrieval_hair_gender_clip-ViT-L-14_sem-b.yml
python main_eval.py --config config/celeba_retrieval_hair_gender_clip-ViT-L-14_zsdebias-seed0.yml
python main_eval.py --config config/celeba_retrieval_hair_gender_clip-ViT-L-14_zsdebias-seed1.yml
python main_eval.py --config config/celeba_retrieval_hair_gender_clip-ViT-L-14_zsdebias-seed2.yml

##### Bias prompts + input-specific prompts (average over 3 seeds for PRISM)
python main_eval.py --config config/celeba_retrieval_hair_gender_clip-ViT-L-14_orth-cali.yml
python main_eval.py --config config/celeba_retrieval_hair_gender_clip-ViT-L-14_sem-bi.yml
python main_eval.py --config config/celeba_retrieval_hair_gender_clip-ViT-L-14_prism-seed0.yml
python main_eval.py --config config/celeba_retrieval_hair_gender_clip-ViT-L-14_prism-seed1.yml
python main_eval.py --config config/celeba_retrieval_hair_gender_clip-ViT-L-14_prism-seed2.yml

##### Bias prompts + input-specific prompts + labeled images
python main_eval.py --config config/celeba_retrieval_hair_gender_clip-ViT-L-14_bendvlm.yml
python main_eval.py --config config/celeba_retrieval_hair_gender_clip-ViT-L-14_bendsem-bi.yml



# --- ResNet backbones (Supp. Mat.) --- #



# Retrieval results
## FairFace
### RN50
#### Race bias
##### Base CLIP
python main_eval.py --config config/fairface_retrieval_stereotype_race_clip-RN50_vanilla.yml

##### Bias-agnostic + input-specific prompts
python main_eval.py --config config/fairface_retrieval_stereotype_race_clip-RN50_roboshot.yml
python main_eval.py --config config/fairface_retrieval_stereotype_race_clip-RN50_sem-i.yml

##### Bias prompts only
python main_eval.py --config config/fairface_retrieval_stereotype_race_clip-RN50_orth-proj.yml
python main_eval.py --config config/fairface_retrieval_stereotype_race_clip-RN50_prism-mini.yml
python main_eval.py --config config/fairface_retrieval_stereotype_race_clip-RN50_sem-b.yml
python main_eval.py --config config/fairface_retrieval_stereotype_race_clip-RN50_zsdebias-seed0.yml
python main_eval.py --config config/fairface_retrieval_stereotype_race_clip-RN50_zsdebias-seed1.yml
python main_eval.py --config config/fairface_retrieval_stereotype_race_clip-RN50_zsdebias-seed2.yml

##### Bias prompts + input-specific prompts (average over 3 seeds for PRISM)
python main_eval.py --config config/fairface_retrieval_stereotype_race_clip-RN50_orth-cali.yml
python main_eval.py --config config/fairface_retrieval_stereotype_race_clip-RN50_sem-bi.yml
python main_eval.py --config config/fairface_retrieval_stereotype_race_clip-RN50_prism-seed0.yml
python main_eval.py --config config/fairface_retrieval_stereotype_race_clip-RN50_prism-seed1.yml
python main_eval.py --config config/fairface_retrieval_stereotype_race_clip-RN50_prism-seed2.yml

##### Bias prompts + input-specific prompts + labeled images
python main_eval.py --config config/fairface_retrieval_stereotype_race_clip-RN50_bendvlm.yml
python main_eval.py --config config/fairface_retrieval_stereotype_race_clip-RN50_bendsem-bi.yml


#### Gender bias
##### Base CLIP
python main_eval.py --config config/fairface_retrieval_stereotype_gender_clip-RN50_vanilla.yml

##### Bias-agnostic + input-specific prompts
python main_eval.py --config config/fairface_retrieval_stereotype_gender_clip-RN50_roboshot.yml
python main_eval.py --config config/fairface_retrieval_stereotype_gender_clip-RN50_sem-i.yml

##### Bias prompts only
python main_eval.py --config config/fairface_retrieval_stereotype_gender_clip-RN50_orth-proj.yml
python main_eval.py --config config/fairface_retrieval_stereotype_gender_clip-RN50_prism-mini.yml
python main_eval.py --config config/fairface_retrieval_stereotype_gender_clip-RN50_sem-b.yml
python main_eval.py --config config/fairface_retrieval_stereotype_gender_clip-RN50_zsdebias-seed0.yml
python main_eval.py --config config/fairface_retrieval_stereotype_gender_clip-RN50_zsdebias-seed1.yml
python main_eval.py --config config/fairface_retrieval_stereotype_gender_clip-RN50_zsdebias-seed2.yml

##### Bias prompts + input-specific prompts (average over 3 seeds for PRISM)
python main_eval.py --config config/fairface_retrieval_stereotype_gender_clip-RN50_orth-cali.yml
python main_eval.py --config config/fairface_retrieval_stereotype_gender_clip-RN50_sem-bi.yml
python main_eval.py --config config/fairface_retrieval_stereotype_gender_clip-RN50_prism-seed0.yml
python main_eval.py --config config/fairface_retrieval_stereotype_gender_clip-RN50_prism-seed1.yml
python main_eval.py --config config/fairface_retrieval_stereotype_gender_clip-RN50_prism-seed2.yml

##### Bias prompts + input-specific prompts + labeled images
python main_eval.py --config config/fairface_retrieval_stereotype_gender_clip-RN50_bendvlm.yml
python main_eval.py --config config/fairface_retrieval_stereotype_gender_clip-RN50_bendsem-bi.yml



### RN101
#### Race bias
##### Base CLIP
python main_eval.py --config config/fairface_retrieval_stereotype_race_clip-RN101_vanilla.yml

##### Bias-agnostic + input-specific prompts
python main_eval.py --config config/fairface_retrieval_stereotype_race_clip-RN101_roboshot.yml
python main_eval.py --config config/fairface_retrieval_stereotype_race_clip-RN101_sem-i.yml

##### Bias prompts only
python main_eval.py --config config/fairface_retrieval_stereotype_race_clip-RN101_orth-proj.yml
python main_eval.py --config config/fairface_retrieval_stereotype_race_clip-RN101_prism-mini.yml
python main_eval.py --config config/fairface_retrieval_stereotype_race_clip-RN101_sem-b.yml
python main_eval.py --config config/fairface_retrieval_stereotype_race_clip-RN101_zsdebias-seed0.yml
python main_eval.py --config config/fairface_retrieval_stereotype_race_clip-RN101_zsdebias-seed1.yml
python main_eval.py --config config/fairface_retrieval_stereotype_race_clip-RN101_zsdebias-seed2.yml

##### Bias prompts + input-specific prompts (average over 3 seeds for PRISM)
python main_eval.py --config config/fairface_retrieval_stereotype_race_clip-RN101_orth-cali.yml
python main_eval.py --config config/fairface_retrieval_stereotype_race_clip-RN101_sem-bi.yml
python main_eval.py --config config/fairface_retrieval_stereotype_race_clip-RN101_prism-seed0.yml
python main_eval.py --config config/fairface_retrieval_stereotype_race_clip-RN101_prism-seed1.yml
python main_eval.py --config config/fairface_retrieval_stereotype_race_clip-RN101_prism-seed2.yml

##### Bias prompts + input-specific prompts + labeled images
python main_eval.py --config config/fairface_retrieval_stereotype_race_clip-RN101_bendvlm.yml
python main_eval.py --config config/fairface_retrieval_stereotype_race_clip-RN101_bendsem-bi.yml


#### Gender bias
##### Base CLIP
python main_eval.py --config config/fairface_retrieval_stereotype_gender_clip-RN101_vanilla.yml

##### Bias-agnostic + input-specific prompts
python main_eval.py --config config/fairface_retrieval_stereotype_gender_clip-RN101_roboshot.yml
python main_eval.py --config config/fairface_retrieval_stereotype_gender_clip-RN101_sem-i.yml

##### Bias prompts only
python main_eval.py --config config/fairface_retrieval_stereotype_gender_clip-RN101_orth-proj.yml
python main_eval.py --config config/fairface_retrieval_stereotype_gender_clip-RN101_prism-mini.yml
python main_eval.py --config config/fairface_retrieval_stereotype_gender_clip-RN101_sem-b.yml
python main_eval.py --config config/fairface_retrieval_stereotype_gender_clip-RN101_zsdebias-seed0.yml
python main_eval.py --config config/fairface_retrieval_stereotype_gender_clip-RN101_zsdebias-seed1.yml
python main_eval.py --config config/fairface_retrieval_stereotype_gender_clip-RN101_zsdebias-seed2.yml

##### Bias prompts + input-specific prompts (average over 3 seeds for PRISM)
python main_eval.py --config config/fairface_retrieval_stereotype_gender_clip-RN101_orth-cali.yml
python main_eval.py --config config/fairface_retrieval_stereotype_gender_clip-RN101_sem-bi.yml
python main_eval.py --config config/fairface_retrieval_stereotype_gender_clip-RN101_prism-seed0.yml
python main_eval.py --config config/fairface_retrieval_stereotype_gender_clip-RN101_prism-seed1.yml
python main_eval.py --config config/fairface_retrieval_stereotype_gender_clip-RN101_prism-seed2.yml

##### Bias prompts + input-specific prompts + labeled images
python main_eval.py --config config/fairface_retrieval_stereotype_gender_clip-RN101_bendvlm.yml
python main_eval.py --config config/fairface_retrieval_stereotype_gender_clip-RN101_bendsem-bi.yml




## UTKFace
### RN50
#### Race bias
##### Base CLIP
python main_eval.py --config config/utkface_retrieval_stereotype_race_clip-RN50_vanilla.yml

##### Bias-agnostic + input-specific prompts
python main_eval.py --config config/utkface_retrieval_stereotype_race_clip-RN50_roboshot.yml
python main_eval.py --config config/utkface_retrieval_stereotype_race_clip-RN50_sem-i.yml

##### Bias prompts only
python main_eval.py --config config/utkface_retrieval_stereotype_race_clip-RN50_orth-proj.yml
python main_eval.py --config config/utkface_retrieval_stereotype_race_clip-RN50_prism-mini.yml
python main_eval.py --config config/utkface_retrieval_stereotype_race_clip-RN50_sem-b.yml
python main_eval.py --config config/utkface_retrieval_stereotype_race_clip-RN50_zsdebias-seed0.yml
python main_eval.py --config config/utkface_retrieval_stereotype_race_clip-RN50_zsdebias-seed1.yml
python main_eval.py --config config/utkface_retrieval_stereotype_race_clip-RN50_zsdebias-seed2.yml

##### Bias prompts + input-specific prompts (average over 3 seeds for PRISM)
python main_eval.py --config config/utkface_retrieval_stereotype_race_clip-RN50_orth-cali.yml
python main_eval.py --config config/utkface_retrieval_stereotype_race_clip-RN50_sem-bi.yml
python main_eval.py --config config/utkface_retrieval_stereotype_race_clip-RN50_prism-seed0.yml
python main_eval.py --config config/utkface_retrieval_stereotype_race_clip-RN50_prism-seed1.yml
python main_eval.py --config config/utkface_retrieval_stereotype_race_clip-RN50_prism-seed2.yml

##### Bias prompts + input-specific prompts + labeled images
python main_eval.py --config config/utkface_retrieval_stereotype_race_clip-RN50_bendvlm.yml
python main_eval.py --config config/utkface_retrieval_stereotype_race_clip-RN50_bendsem-bi.yml


#### Gender bias
##### Base CLIP
python main_eval.py --config config/utkface_retrieval_stereotype_gender_clip-RN50_vanilla.yml

##### Bias-agnostic + input-specific prompts
python main_eval.py --config config/utkface_retrieval_stereotype_gender_clip-RN50_roboshot.yml
python main_eval.py --config config/utkface_retrieval_stereotype_gender_clip-RN50_sem-i.yml

##### Bias prompts only
python main_eval.py --config config/utkface_retrieval_stereotype_gender_clip-RN50_orth-proj.yml
python main_eval.py --config config/utkface_retrieval_stereotype_gender_clip-RN50_prism-mini.yml
python main_eval.py --config config/utkface_retrieval_stereotype_gender_clip-RN50_sem-b.yml
python main_eval.py --config config/utkface_retrieval_stereotype_gender_clip-RN50_zsdebias-seed0.yml
python main_eval.py --config config/utkface_retrieval_stereotype_gender_clip-RN50_zsdebias-seed1.yml
python main_eval.py --config config/utkface_retrieval_stereotype_gender_clip-RN50_zsdebias-seed2.yml

##### Bias prompts + input-specific prompts (average over 3 seeds for PRISM)
python main_eval.py --config config/utkface_retrieval_stereotype_gender_clip-RN50_orth-cali.yml
python main_eval.py --config config/utkface_retrieval_stereotype_gender_clip-RN50_sem-bi.yml
python main_eval.py --config config/utkface_retrieval_stereotype_gender_clip-RN50_prism-seed0.yml
python main_eval.py --config config/utkface_retrieval_stereotype_gender_clip-RN50_prism-seed1.yml
python main_eval.py --config config/utkface_retrieval_stereotype_gender_clip-RN50_prism-seed2.yml

##### Bias prompts + input-specific prompts + labeled images
python main_eval.py --config config/utkface_retrieval_stereotype_gender_clip-RN50_bendvlm.yml
python main_eval.py --config config/utkface_retrieval_stereotype_gender_clip-RN50_bendsem-bi.yml



### RN101
#### Race bias
##### Base CLIP
python main_eval.py --config config/utkface_retrieval_stereotype_race_clip-RN101_vanilla.yml

##### Bias-agnostic + input-specific prompts
python main_eval.py --config config/utkface_retrieval_stereotype_race_clip-RN101_roboshot.yml
python main_eval.py --config config/utkface_retrieval_stereotype_race_clip-RN101_sem-i.yml

##### Bias prompts only
python main_eval.py --config config/utkface_retrieval_stereotype_race_clip-RN101_orth-proj.yml
python main_eval.py --config config/utkface_retrieval_stereotype_race_clip-RN101_prism-mini.yml
python main_eval.py --config config/utkface_retrieval_stereotype_race_clip-RN101_sem-b.yml
python main_eval.py --config config/utkface_retrieval_stereotype_race_clip-RN101_zsdebias-seed0.yml
python main_eval.py --config config/utkface_retrieval_stereotype_race_clip-RN101_zsdebias-seed1.yml
python main_eval.py --config config/utkface_retrieval_stereotype_race_clip-RN101_zsdebias-seed2.yml

##### Bias prompts + input-specific prompts (average over 3 seeds for PRISM)
python main_eval.py --config config/utkface_retrieval_stereotype_race_clip-RN101_orth-cali.yml
python main_eval.py --config config/utkface_retrieval_stereotype_race_clip-RN101_sem-bi.yml
python main_eval.py --config config/utkface_retrieval_stereotype_race_clip-RN101_prism-seed0.yml
python main_eval.py --config config/utkface_retrieval_stereotype_race_clip-RN101_prism-seed1.yml
python main_eval.py --config config/utkface_retrieval_stereotype_race_clip-RN101_prism-seed2.yml

##### Bias prompts + input-specific prompts + labeled images
python main_eval.py --config config/utkface_retrieval_stereotype_race_clip-RN101_bendvlm.yml
python main_eval.py --config config/utkface_retrieval_stereotype_race_clip-RN101_bendsem-bi.yml


#### Gender bias
##### Base CLIP
python main_eval.py --config config/utkface_retrieval_stereotype_gender_clip-RN101_vanilla.yml

##### Bias-agnostic + input-specific prompts
python main_eval.py --config config/utkface_retrieval_stereotype_gender_clip-RN101_roboshot.yml
python main_eval.py --config config/utkface_retrieval_stereotype_gender_clip-RN101_sem-i.yml

##### Bias prompts only
python main_eval.py --config config/utkface_retrieval_stereotype_gender_clip-RN101_orth-proj.yml
python main_eval.py --config config/utkface_retrieval_stereotype_gender_clip-RN101_prism-mini.yml
python main_eval.py --config config/utkface_retrieval_stereotype_gender_clip-RN101_sem-b.yml
python main_eval.py --config config/utkface_retrieval_stereotype_gender_clip-RN101_zsdebias-seed0.yml
python main_eval.py --config config/utkface_retrieval_stereotype_gender_clip-RN101_zsdebias-seed1.yml
python main_eval.py --config config/utkface_retrieval_stereotype_gender_clip-RN101_zsdebias-seed2.yml

##### Bias prompts + input-specific prompts (average over 3 seeds for PRISM)
python main_eval.py --config config/utkface_retrieval_stereotype_gender_clip-RN101_orth-cali.yml
python main_eval.py --config config/utkface_retrieval_stereotype_gender_clip-RN101_sem-bi.yml
python main_eval.py --config config/utkface_retrieval_stereotype_gender_clip-RN101_prism-seed0.yml
python main_eval.py --config config/utkface_retrieval_stereotype_gender_clip-RN101_prism-seed1.yml
python main_eval.py --config config/utkface_retrieval_stereotype_gender_clip-RN101_prism-seed2.yml

##### Bias prompts + input-specific prompts + labeled images
python main_eval.py --config config/utkface_retrieval_stereotype_gender_clip-RN101_bendvlm.yml
python main_eval.py --config config/utkface_retrieval_stereotype_gender_clip-RN101_bendsem-bi.yml





# Zero-shot classification results
## CelebA (gender bias)
### RN50
#### Base CLIP
python main_eval.py --config config/celeba_zs-classification_celeba-blond-hair-gender_clip-RN50_vanilla.yml

#### Bias-agnostic + input-specific prompts
python main_eval.py --config config/celeba_zs-classification_celeba-blond-hair-gender_clip-RN50_roboshot.yml
python main_eval.py --config config/celeba_zs-classification_celeba-blond-hair-gender_clip-RN50_sem-i.yml

#### Bias prompts only
python main_eval.py --config config/celeba_zs-classification_celeba-blond-hair-gender_clip-RN50_orth-proj.yml
python main_eval.py --config config/celeba_zs-classification_celeba-blond-hair-gender_clip-RN50_prism-mini.yml
python main_eval.py --config config/celeba_zs-classification_celeba-blond-hair-gender_clip-RN50_sem-b.yml
python main_eval.py --config config/celeba_zs-classification_celeba-blond-hair-gender_clip-RN50_zsdebias-seed0.yml
python main_eval.py --config config/celeba_zs-classification_celeba-blond-hair-gender_clip-RN50_zsdebias-seed1.yml
python main_eval.py --config config/celeba_zs-classification_celeba-blond-hair-gender_clip-RN50_zsdebias-seed2.yml

#### Bias prompts + input-specific prompts (average over 3 seeds for PRISM)
python main_eval.py --config config/celeba_zs-classification_celeba-blond-hair-gender_clip-RN50_orth-cali.yml
python main_eval.py --config config/celeba_zs-classification_celeba-blond-hair-gender_clip-RN50_sem-bi.yml
python main_eval.py --config config/celeba_zs-classification_celeba-blond-hair-gender_clip-RN50_prism-seed0.yml
python main_eval.py --config config/celeba_zs-classification_celeba-blond-hair-gender_clip-RN50_prism-seed1.yml
python main_eval.py --config config/celeba_zs-classification_celeba-blond-hair-gender_clip-RN50_prism-seed2.yml

#### Bias prompts + input-specific prompts + labeled images
python main_eval.py --config config/celeba_zs-classification_celeba-blond-hair-gender_clip-RN50_bendvlm.yml
python main_eval.py --config config/celeba_zs-classification_celeba-blond-hair-gender_clip-RN50_bendsem-bi.yml


### RN101
#### Base CLIP
python main_eval.py --config config/celeba_zs-classification_celeba-blond-hair-gender_clip-RN101_vanilla.yml

#### Bias-agnostic + input-specific prompts
python main_eval.py --config config/celeba_zs-classification_celeba-blond-hair-gender_clip-RN101_roboshot.yml
python main_eval.py --config config/celeba_zs-classification_celeba-blond-hair-gender_clip-RN101_sem-i.yml

#### Bias prompts only
python main_eval.py --config config/celeba_zs-classification_celeba-blond-hair-gender_clip-RN101_orth-proj.yml
python main_eval.py --config config/celeba_zs-classification_celeba-blond-hair-gender_clip-RN101_prism-mini.yml
python main_eval.py --config config/celeba_zs-classification_celeba-blond-hair-gender_clip-RN101_sem-b.yml
python main_eval.py --config config/celeba_zs-classification_celeba-blond-hair-gender_clip-RN101_zsdebias-seed0.yml
python main_eval.py --config config/celeba_zs-classification_celeba-blond-hair-gender_clip-RN101_zsdebias-seed1.yml
python main_eval.py --config config/celeba_zs-classification_celeba-blond-hair-gender_clip-RN101_zsdebias-seed2.yml

#### Bias prompts + input-specific prompts (average over 3 seeds for PRISM)
python main_eval.py --config config/celeba_zs-classification_celeba-blond-hair-gender_clip-RN101_orth-cali.yml
python main_eval.py --config config/celeba_zs-classification_celeba-blond-hair-gender_clip-RN101_sem-bi.yml
python main_eval.py --config config/celeba_zs-classification_celeba-blond-hair-gender_clip-RN101_prism-seed0.yml
python main_eval.py --config config/celeba_zs-classification_celeba-blond-hair-gender_clip-RN101_prism-seed1.yml
python main_eval.py --config config/celeba_zs-classification_celeba-blond-hair-gender_clip-RN101_prism-seed2.yml

#### Bias prompts + input-specific prompts + labeled images
python main_eval.py --config config/celeba_zs-classification_celeba-blond-hair-gender_clip-RN101_bendvlm.yml
python main_eval.py --config config/celeba_zs-classification_celeba-blond-hair-gender_clip-RN101_bendsem-bi.yml



## Waterbirds (background bias)
### RN50
#### Base CLIP
python main_eval.py --config config/waterbirds_zs-classification_waterbirds_clip-RN50_vanilla.yml

#### Bias-agnostic + input-specific prompts
python main_eval.py --config config/waterbirds_zs-classification_waterbirds_clip-RN50_roboshot.yml
python main_eval.py --config config/waterbirds_zs-classification_waterbirds_clip-RN50_sem-i.yml

#### Bias prompts only
python main_eval.py --config config/waterbirds_zs-classification_waterbirds_clip-RN50_orth-proj.yml
python main_eval.py --config config/waterbirds_zs-classification_waterbirds_clip-RN50_prism-mini.yml
python main_eval.py --config config/waterbirds_zs-classification_waterbirds_clip-RN50_sem-b.yml
python main_eval.py --config config/waterbirds_zs-classification_waterbirds_clip-RN50_zsdebias-seed0.yml
python main_eval.py --config config/waterbirds_zs-classification_waterbirds_clip-RN50_zsdebias-seed1.yml
python main_eval.py --config config/waterbirds_zs-classification_waterbirds_clip-RN50_zsdebias-seed2.yml

#### Bias prompts + input-specific prompts (average over 3 seeds for PRISM)
python main_eval.py --config config/waterbirds_zs-classification_waterbirds_clip-RN50_orth-cali.yml
python main_eval.py --config config/waterbirds_zs-classification_waterbirds_clip-RN50_sem-bi.yml
python main_eval.py --config config/waterbirds_zs-classification_waterbirds_clip-RN50_prism-seed0.yml
python main_eval.py --config config/waterbirds_zs-classification_waterbirds_clip-RN50_prism-seed1.yml
python main_eval.py --config config/waterbirds_zs-classification_waterbirds_clip-RN50_prism-seed2.yml

#### Bias prompts + input-specific prompts + labeled images
python main_eval.py --config config/waterbirds_zs-classification_waterbirds_clip-RN50_bendvlm.yml
python main_eval.py --config config/waterbirds_zs-classification_waterbirds_clip-RN50_bendsem-bi.yml


### RN101
#### Base CLIP
python main_eval.py --config config/waterbirds_zs-classification_waterbirds_clip-RN101_vanilla.yml

#### Bias-agnostic + input-specific prompts
python main_eval.py --config config/waterbirds_zs-classification_waterbirds_clip-RN101_roboshot.yml
python main_eval.py --config config/waterbirds_zs-classification_waterbirds_clip-RN101_sem-i.yml

#### Bias prompts only
python main_eval.py --config config/waterbirds_zs-classification_waterbirds_clip-RN101_orth-proj.yml
python main_eval.py --config config/waterbirds_zs-classification_waterbirds_clip-RN101_prism-mini.yml
python main_eval.py --config config/waterbirds_zs-classification_waterbirds_clip-RN101_sem-b.yml
python main_eval.py --config config/waterbirds_zs-classification_waterbirds_clip-RN101_zsdebias-seed0.yml
python main_eval.py --config config/waterbirds_zs-classification_waterbirds_clip-RN101_zsdebias-seed1.yml
python main_eval.py --config config/waterbirds_zs-classification_waterbirds_clip-RN101_zsdebias-seed2.yml

#### Bias prompts + input-specific prompts (average over 3 seeds for PRISM)
python main_eval.py --config config/waterbirds_zs-classification_waterbirds_clip-RN101_orth-cali.yml
python main_eval.py --config config/waterbirds_zs-classification_waterbirds_clip-RN101_sem-bi.yml
python main_eval.py --config config/waterbirds_zs-classification_waterbirds_clip-RN101_prism-seed0.yml
python main_eval.py --config config/waterbirds_zs-classification_waterbirds_clip-RN101_prism-seed1.yml
python main_eval.py --config config/waterbirds_zs-classification_waterbirds_clip-RN101_prism-seed2.yml

#### Bias prompts + input-specific prompts + labeled images
python main_eval.py --config config/waterbirds_zs-classification_waterbirds_clip-RN101_bendvlm.yml
python main_eval.py --config config/waterbirds_zs-classification_waterbirds_clip-RN101_bendsem-bi.yml





# Additional retrieval results on CelebA
## RN50
### Stereotype queries
##### Base CLIP
python main_eval.py --config config/celeba_retrieval_stereotype_gender_clip-RN50_vanilla.yml

##### Bias-agnostic + input-specific prompts
python main_eval.py --config config/celeba_retrieval_stereotype_gender_clip-RN50_roboshot.yml
python main_eval.py --config config/celeba_retrieval_stereotype_gender_clip-RN50_sem-i.yml

##### Bias prompts only
python main_eval.py --config config/celeba_retrieval_stereotype_gender_clip-RN50_orth-proj.yml
python main_eval.py --config config/celeba_retrieval_stereotype_gender_clip-RN50_prism-mini.yml
python main_eval.py --config config/celeba_retrieval_stereotype_gender_clip-RN50_sem-b.yml
python main_eval.py --config config/celeba_retrieval_stereotype_gender_clip-RN50_zsdebias-seed0.yml
python main_eval.py --config config/celeba_retrieval_stereotype_gender_clip-RN50_zsdebias-seed1.yml
python main_eval.py --config config/celeba_retrieval_stereotype_gender_clip-RN50_zsdebias-seed2.yml

##### Bias prompts + input-specific prompts (average over 3 seeds for PRISM)
python main_eval.py --config config/celeba_retrieval_stereotype_gender_clip-RN50_orth-cali.yml
python main_eval.py --config config/celeba_retrieval_stereotype_gender_clip-RN50_sem-bi.yml
python main_eval.py --config config/celeba_retrieval_stereotype_gender_clip-RN50_prism-seed0.yml
python main_eval.py --config config/celeba_retrieval_stereotype_gender_clip-RN50_prism-seed1.yml
python main_eval.py --config config/celeba_retrieval_stereotype_gender_clip-RN50_prism-seed2.yml

##### Bias prompts + input-specific prompts + labeled images
python main_eval.py --config config/celeba_retrieval_stereotype_gender_clip-RN50_bendvlm.yml
python main_eval.py --config config/celeba_retrieval_stereotype_gender_clip-RN50_bendsem-bi.yml


### Hair color queries
##### Base CLIP
python main_eval.py --config config/celeba_retrieval_hair_gender_clip-RN50_vanilla.yml

##### Bias-agnostic + input-specific prompts
python main_eval.py --config config/celeba_retrieval_hair_gender_clip-RN50_roboshot.yml
python main_eval.py --config config/celeba_retrieval_hair_gender_clip-RN50_sem-i.yml

##### Bias prompts only
python main_eval.py --config config/celeba_retrieval_hair_gender_clip-RN50_orth-proj.yml
python main_eval.py --config config/celeba_retrieval_hair_gender_clip-RN50_prism-mini.yml
python main_eval.py --config config/celeba_retrieval_hair_gender_clip-RN50_sem-b.yml
python main_eval.py --config config/celeba_retrieval_hair_gender_clip-RN50_zsdebias-seed0.yml
python main_eval.py --config config/celeba_retrieval_hair_gender_clip-RN50_zsdebias-seed1.yml
python main_eval.py --config config/celeba_retrieval_hair_gender_clip-RN50_zsdebias-seed2.yml

##### Bias prompts + input-specific prompts (average over 3 seeds for PRISM)
python main_eval.py --config config/celeba_retrieval_hair_gender_clip-RN50_orth-cali.yml
python main_eval.py --config config/celeba_retrieval_hair_gender_clip-RN50_sem-bi.yml
python main_eval.py --config config/celeba_retrieval_hair_gender_clip-RN50_prism-seed0.yml
python main_eval.py --config config/celeba_retrieval_hair_gender_clip-RN50_prism-seed1.yml
python main_eval.py --config config/celeba_retrieval_hair_gender_clip-RN50_prism-seed2.yml

##### Bias prompts + input-specific prompts + labeled images
python main_eval.py --config config/celeba_retrieval_hair_gender_clip-RN50_bendvlm.yml
python main_eval.py --config config/celeba_retrieval_hair_gender_clip-RN50_bendsem-bi.yml



## RN101
### Stereotype queries
##### Base CLIP
python main_eval.py --config config/celeba_retrieval_stereotype_gender_clip-RN101_vanilla.yml

##### Bias-agnostic + input-specific prompts
python main_eval.py --config config/celeba_retrieval_stereotype_gender_clip-RN101_roboshot.yml
python main_eval.py --config config/celeba_retrieval_stereotype_gender_clip-RN101_sem-i.yml

##### Bias prompts only
python main_eval.py --config config/celeba_retrieval_stereotype_gender_clip-RN101_orth-proj.yml
python main_eval.py --config config/celeba_retrieval_stereotype_gender_clip-RN101_prism-mini.yml
python main_eval.py --config config/celeba_retrieval_stereotype_gender_clip-RN101_sem-b.yml
python main_eval.py --config config/celeba_retrieval_stereotype_gender_clip-RN101_zsdebias-seed0.yml
python main_eval.py --config config/celeba_retrieval_stereotype_gender_clip-RN101_zsdebias-seed1.yml
python main_eval.py --config config/celeba_retrieval_stereotype_gender_clip-RN101_zsdebias-seed2.yml

##### Bias prompts + input-specific prompts (average over 3 seeds for PRISM)
python main_eval.py --config config/celeba_retrieval_stereotype_gender_clip-RN101_orth-cali.yml
python main_eval.py --config config/celeba_retrieval_stereotype_gender_clip-RN101_sem-bi.yml
python main_eval.py --config config/celeba_retrieval_stereotype_gender_clip-RN101_prism-seed0.yml
python main_eval.py --config config/celeba_retrieval_stereotype_gender_clip-RN101_prism-seed1.yml
python main_eval.py --config config/celeba_retrieval_stereotype_gender_clip-RN101_prism-seed2.yml

##### Bias prompts + input-specific prompts + labeled images
python main_eval.py --config config/celeba_retrieval_stereotype_gender_clip-RN101_bendvlm.yml
python main_eval.py --config config/celeba_retrieval_stereotype_gender_clip-RN101_bendsem-bi.yml


### Hair color queries
##### Base CLIP
python main_eval.py --config config/celeba_retrieval_hair_gender_clip-RN101_vanilla.yml

##### Bias-agnostic + input-specific prompts
python main_eval.py --config config/celeba_retrieval_hair_gender_clip-RN101_roboshot.yml
python main_eval.py --config config/celeba_retrieval_hair_gender_clip-RN101_sem-i.yml

##### Bias prompts only
python main_eval.py --config config/celeba_retrieval_hair_gender_clip-RN101_orth-proj.yml
python main_eval.py --config config/celeba_retrieval_hair_gender_clip-RN101_prism-mini.yml
python main_eval.py --config config/celeba_retrieval_hair_gender_clip-RN101_sem-b.yml
python main_eval.py --config config/celeba_retrieval_hair_gender_clip-RN101_zsdebias-seed0.yml
python main_eval.py --config config/celeba_retrieval_hair_gender_clip-RN101_zsdebias-seed1.yml
python main_eval.py --config config/celeba_retrieval_hair_gender_clip-RN101_zsdebias-seed2.yml

##### Bias prompts + input-specific prompts (average over 3 seeds for PRISM)
python main_eval.py --config config/celeba_retrieval_hair_gender_clip-RN101_orth-cali.yml
python main_eval.py --config config/celeba_retrieval_hair_gender_clip-RN101_sem-bi.yml
python main_eval.py --config config/celeba_retrieval_hair_gender_clip-RN101_prism-seed0.yml
python main_eval.py --config config/celeba_retrieval_hair_gender_clip-RN101_prism-seed1.yml
python main_eval.py --config config/celeba_retrieval_hair_gender_clip-RN101_prism-seed2.yml

##### Bias prompts + input-specific prompts + labeled images
python main_eval.py --config config/celeba_retrieval_hair_gender_clip-RN101_bendvlm.yml
python main_eval.py --config config/celeba_retrieval_hair_gender_clip-RN101_bendsem-bi.yml
