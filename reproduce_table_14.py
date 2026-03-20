import pandas as pd

# 1. Load Data
df = pd.read_feather('all_results.feather')

# 2. Filter for ResNet Table 2 criteria
df_res_zs = df[
    (df['task'] == 'zs_classification') & 
    (df['dataset'].isin(['celeba', 'waterbirds'])) &
    (df['metric'].isin(['Acc.', 'WG Acc.'])) &
    (df['backbone'].isin(['RN50', 'RN101']))  # Filter for ResNets
].copy()

# 3. Average over seeds and specific settings
grouped = df_res_zs.groupby(['dataset', 'backbone', 'metric', 'method'], as_index=False)['mean'].mean()

# 4. Create the Pivot Table
pivot_df = grouped.pivot_table(
    index='method', 
    columns=['dataset', 'backbone', 'metric'], 
    values='mean'
)

# 5. Calculate the 'Gap' metric (Acc. - WG Acc.)
for dataset in ['celeba', 'waterbirds']:
    for backbone in ['RN50', 'RN101']:
        if (dataset, backbone, 'Acc.') in pivot_df.columns and (dataset, backbone, 'WG Acc.') in pivot_df.columns:
            pivot_df[(dataset, backbone, 'Gap')] = pivot_df[(dataset, backbone, 'Acc.')] - pivot_df[(dataset, backbone, 'WG Acc.')]

# 6. Reorder Columns to perfectly match the LaTeX table
datasets = ['celeba', 'waterbirds']
backbones = ['RN50', 'RN101']
metrics = ['Acc.', 'WG Acc.', 'Gap']

col_order = pd.MultiIndex.from_product(
    [datasets, backbones, metrics],
    names=['Dataset', 'Backbone', 'Metric']
)

# Drop any columns that didn't successfully generate
existing_cols = [col for col in col_order if col in pivot_df.columns]
pivot_df = pivot_df.reindex(columns=existing_cols)

# Rename 'WG Acc.' to 'WG' to match the LaTeX headers
pivot_df = pivot_df.rename(columns={'WG Acc.': 'WG'})

# 7. Reorder Rows and rename them to match the paper
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

# 8. Format output to 3 decimal places
pd.options.display.float_format = '{:.3f}'.format
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

print("\n" + "="*100)
print("TABLE 14 (ResNet): Measuring zero-shot classification fairness on CelebA and Waterbirds")
print("="*100)
print(pivot_df)
