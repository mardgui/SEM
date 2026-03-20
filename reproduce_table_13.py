import pandas as pd
import re

# 1. Load Data
df = pd.read_feather('all_results.feather')

# 2. Filter for ResNet Table criteria
df_res = df[
    (df['task'] == 'retrieval') & 
    (df['dataset'].isin(['fairface', 'utkface'])) &
    (df['setting'].str.contains('query=stereotype', na=False)) &
    (df['backbone'].isin(['RN50', 'RN101']))  # Specifically filtering for ResNets
].copy()

# 3. Extract the Bias target
def extract_bias(setting_str):
    match = re.search(r'bias=([^|]+)', str(setting_str))
    if match:
        val = match.group(1)
        return 'race' if val == 'utkrace' else val
    return None

df_res['bias'] = df_res['setting'].apply(extract_bias)

# 4. Average over seeds
grouped = df_res.groupby(['dataset', 'backbone', 'bias', 'metric', 'method'], as_index=False)['mean'].mean()

# 5. Create the Pivot Table
pivot_df = grouped.pivot_table(
    index='method', 
    columns=['dataset', 'backbone', 'bias', 'metric'], 
    values='mean'
)

# 6. Reorder Columns to match the target LaTeX table
datasets = ['fairface', 'utkface']
backbones = ['RN50', 'RN101']
biases = ['race', 'gender']
metrics = ['kl', 'max_skew']

col_order = pd.MultiIndex.from_product(
    [datasets, backbones, biases, metrics],
    names=['Dataset', 'Backbone', 'Bias', 'Metric']
)

# Safe reindexing in case some methods/metrics failed to run
existing_cols = [col for col in col_order if col in pivot_df.columns]
pivot_df = pivot_df.reindex(columns=existing_cols)

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

print("\n" + "="*80)
print("TABLE 13 (ResNet): Measuring race and gender bias for stereotype queries")
print("="*80)
print(pivot_df)
