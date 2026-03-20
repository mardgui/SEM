import pandas as pd

# 1. Load Data
df = pd.read_feather('all_results.feather')

# 2. Filter for Table 9 criteria (CelebA Retrieval only)
df_t9 = df[
    (df['task'] == 'retrieval') & 
    (df['dataset'] == 'celeba')
].copy()

# 3. Standardize Backbone names
df_t9['backbone'] = df_t9['backbone'].replace({
    'ViT-L-14': 'ViT-L/14@336px',
    'ViT-L-14@336px': 'ViT-L/14@336px'
})
df_t9 = df_t9[df_t9['backbone'].isin(['ViT-B-16', 'ViT-L/14@336px'])]

# 4. Create the 'Query Type' column based on the setting string
def get_query_type(setting_str):
    setting_str = str(setting_str).lower()
    if 'stereotype' in setting_str:
        return 'Stereotype'
    elif 'hair' in setting_str:
        return 'Hair Color'
    return None

df_t9['query_type'] = df_t9['setting'].apply(get_query_type)
df_t9 = df_t9[df_t9['query_type'].notna()]  # Drop anything that isn't Stereotype or Hair Color

# 5. Standardize Metric names to match LaTeX
df_t9['metric'] = df_t9['metric'].replace({
    'kl': 'KL',
    'max_skew': 'MS',
    'precision': 'Prec.'
})

# 6. Average over seeds and specific hair color settings (black, blond, brown, gray)
grouped = df_t9.groupby(['backbone', 'query_type', 'metric', 'method'], as_index=False)['mean'].mean()

# 7. Create the Pivot Table
pivot_df = grouped.pivot_table(
    index='method', 
    columns=['backbone', 'query_type', 'metric'], 
    values='mean'
)

# 8. Reorder Columns to perfectly match the LaTeX table
col_order_tuples = [
    ('ViT-B-16', 'Stereotype', 'KL'),
    ('ViT-B-16', 'Stereotype', 'MS'),
    ('ViT-B-16', 'Hair Color', 'KL'),
    ('ViT-B-16', 'Hair Color', 'MS'),
    ('ViT-B-16', 'Hair Color', 'Prec.'),
    ('ViT-L/14@336px', 'Stereotype', 'KL'),
    ('ViT-L/14@336px', 'Stereotype', 'MS'),
    ('ViT-L/14@336px', 'Hair Color', 'KL'),
    ('ViT-L/14@336px', 'Hair Color', 'MS'),
    ('ViT-L/14@336px', 'Hair Color', 'Prec.')
]

col_order = pd.MultiIndex.from_tuples(
    col_order_tuples, 
    names=['Backbone', 'Query', 'Metric']
)

# Reindex and drop any columns that might be missing (prevents KeyErrors)
existing_cols = [col for col in col_order if col in pivot_df.columns]
pivot_df = pivot_df.reindex(columns=existing_cols)

# 9. Reorder Rows and rename them to match the paper (ADDED ZSDEBIAS)
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

# 10. Format output to 3 decimal places
pd.options.display.float_format = '{:.3f}'.format
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

print("\n" + "="*110)
print("TABLE 9: Measuring gender bias for stereotype and hair color queries on CelebA")
print("="*110)
print(pivot_df)
