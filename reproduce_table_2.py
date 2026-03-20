import pandas as pd
import re

# 1. Load Data
df = pd.read_feather('all_results.feather')

# 2. Filter for Table 1 criteria
df_t1 = df[
    (df['task'] == 'retrieval') & 
    (df['dataset'].isin(['fairface', 'utkface'])) &
    (df['setting'].str.contains('query=stereotype', na=False))
].copy()

# 3. Extract the Bias target from the setting string
# This looks for "bias=race", "bias=gender", or "bias=utkrace"
def extract_bias(setting_str):
    match = re.search(r'bias=([^|]+)', str(setting_str))
    if match:
        val = match.group(1)
        # Map utkrace to race if UTKFace uses a different string
        return 'race' if val == 'utkrace' else val
    return None

df_t1['bias'] = df_t1['setting'].apply(extract_bias)

# 4. Standardize Backbone names just in case they differ slightly in JSON
df_t1['backbone'] = df_t1['backbone'].replace({
    'ViT-L-14': 'ViT-L/14@336px',
    'ViT-L-14@336px': 'ViT-L/14@336px'
})

# Filter out any backbones that aren't for Table 1 (e.g., ResNets)
df_t1 = df_t1[df_t1['backbone'].isin(['ViT-B-16', 'ViT-L/14@336px'])]

# 5. Average over seeds (Groups by everything EXCEPT seed, taking the mean of 'mean')
grouped = df_t1.groupby(['dataset', 'backbone', 'bias', 'metric', 'method'], as_index=False)['mean'].mean()

# 6. Create the Pivot Table
pivot_df = grouped.pivot_table(
    index='method', 
    columns=['dataset', 'backbone', 'bias', 'metric'], 
    values='mean'
)

# 7. Reorder Columns to perfectly match the LaTeX table (left to right)
datasets = ['fairface', 'utkface']
backbones = ['ViT-B-16', 'ViT-L/14@336px']
biases = ['race', 'gender']
metrics = ['kl', 'max_skew']

# Build the exact MultiIndex order expected
col_order = pd.MultiIndex.from_product(
    [datasets, backbones, biases, metrics],
    names=['Dataset', 'Backbone', 'Bias', 'Metric']
)
pivot_df = pivot_df.reindex(columns=col_order)

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

print("\n" + "="*80)
print("TABLE 2: Measuring race and gender bias for stereotype queries")
print("="*80)
print(pivot_df)
