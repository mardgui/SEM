#!/bin/bash

# Precompute for Ethnicity
python precompute_dataset.py --main_task "profession" --bias_attr "ethnicity" --clip_model "ViT-B/16"
python precompute_dataset.py --main_task "profession" --bias_attr "ethnicity" --clip_model "ViT-L/14@336px"

# Precompute for Gender
python precompute_dataset.py --main_task "profession" --bias_attr "gender" --clip_model "ViT-B/16"
python precompute_dataset.py --main_task "profession" --bias_attr "gender" --clip_model "ViT-L/14@336px"
