#!/bin/bash

# Evaluate Ethnicity
python disentanglement_experiment.py --main_task "profession" --bias_attr "ethnicity" --clip_model "ViT-B/16" --latent "clip"
python disentanglement_experiment.py --main_task "profession" --bias_attr "ethnicity" --clip_model "ViT-B/16" --latent "sae"
python disentanglement_experiment.py --main_task "profession" --bias_attr "ethnicity" --clip_model "ViT-L/14@336px" --latent "clip"
python disentanglement_experiment.py --main_task "profession" --bias_attr "ethnicity" --clip_model "ViT-L/14@336px" --latent "sae"

# Evaluate Gender
python disentanglement_experiment.py --main_task "profession" --bias_attr "gender" --clip_model "ViT-B/16" --latent "clip"
python disentanglement_experiment.py --main_task "profession" --bias_attr "gender" --clip_model "ViT-B/16" --latent "sae"
python disentanglement_experiment.py --main_task "profession" --bias_attr "gender" --clip_model "ViT-L/14@336px" --latent "clip"
python disentanglement_experiment.py --main_task "profession" --bias_attr "gender" --clip_model "ViT-L/14@336px" --latent "sae"
