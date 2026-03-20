import pandas as pd
import numpy as np

# 1. Load Data
df = pd.read_feather('all_results.feather')

# 2. Filter for Table 2 criteria
df_t2 = df[
    (df['task'] == 'zs_classification') & 
    (df['dataset'].isin(['celeba', 'waterbirds'])) &
    (df['metric'].isin(['Acc.', 'WG Acc.']))
].copy()

# 3. Standardize Backbone names
df_t2['backbone'] = df_t2['backbone'].replace({
    'ViT-L-14': 'ViT-L/14@336px',
    'ViT-L-14@336px': 'ViT-L/14@336px'
})

# Filter out ResNets
df_t2 = df_t2[df_t2['backbone'].isin(['ViT-B-16', 'ViT-L/14@336px'])]

# 4. Average over seeds and specific settings (e.g., averaging blond/black/brown hair for CelebA)
grouped = df_t2.groupby(['dataset', 'backbone', 'metric', 'method'], as_index=False)['mean'].mean()

# 5. Create the Pivot Table
pivot_df = grouped.pivot_table(
    index='method', 
    columns=['dataset', 'backbone', 'metric'], 
    values='mean'
)

# 6. Calculate the 'Gap' metric (Acc. - WG Acc.)
# We iterate over the existing dataset and backbone combinations to compute this
for dataset in ['celeba', 'waterbirds']:
    for backbone in ['ViT-B-16', 'ViT-L/14@336px']:
        # Only calculate if both Acc and WG exist for this combo
        if (dataset, backbone, 'Acc.') in pivot_df.columns and (dataset, backbone, 'WG Acc.') in pivot_df.columns:
            pivot_df[(dataset, backbone, 'Gap')] = pivot_df[(dataset, backbone, 'Acc.')] - pivot_df[(dataset, backbone, 'WG Acc.')]

# 7. Reorder Columns to perfectly match the LaTeX table
datasets = ['celeba', 'waterbirds']
backbones = ['ViT-B-16', 'ViT-L/14@336px']
metrics = ['Acc.', 'WG Acc.', 'Gap']

col_order = pd.MultiIndex.from_product(
    [datasets, backbones, metrics],
    names=['Dataset', 'Backbone', 'Metric']
)

# Drop any columns that didn't successfully generate (avoids KeyError)
existing_cols = [col for col in col_order if col in pivot_df.columns]
pivot_df = pivot_df.reindex(columns=existing_cols)

# Rename 'WG Acc.' to 'WG' to match the LaTeX headers
pivot_df = pivot_df.rename(columns={'WG Acc.': 'WG'})

# 8. Reorder Rows and rename them to match the paper
method_order = [
    'vanilla', 
    'roboshot', 'sem-i', 
    'orth-proj', 'prism-mini', 'sem-b', 'zsdebias', 
    'orth-cali', 'sem-bi', 'prism', 
    'bendvlm', 'bendsem-bi'
]

display_names = {
    'vanilla': 'Base CLIP',
    'roboshot': 'RoboShot',
    'sem-i': 'SEM_i',
    'orth-proj': 'Orth-Proj',
    'prism-mini': 'PRISM-mini',
    'sem-b': 'SEM_b',
    'zsdebias': 'ZSDebias',
    'orth-cali': 'Orth-Cali',
    'sem-bi': 'SEM_bi',
    'prism': 'PRISM',
    'bendvlm': 'BendVLM',
    'bendsem-bi': 'BendSEM_bi'
}

pivot_df = pivot_df.reindex(method_order)
pivot_df.index = pivot_df.index.map(display_names)
pivot_df.index.name = 'Method'

# 9. Format output to 3 decimal places
pd.options.display.float_format = '{:.3f}'.format
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

print("\n" + "="*100)
print("TABLE 3: Measuring zero-shot classification fairness on CelebA and Waterbirds")
print("="*100)
print(pivot_df)
