import pandas as pd

# 1. Load Data
df = pd.read_feather('all_results.feather')

# 2. Filter for ResNet Table 9 criteria (CelebA Retrieval only)
df_res_t9 = df[
    (df['task'] == 'retrieval') & 
    (df['dataset'] == 'celeba') &
    (df['backbone'].isin(['RN50', 'RN101']))  # Filter for ResNets
].copy()

# 3. Create the 'Query Type' column based on the setting string
def get_query_type(setting_str):
    setting_str = str(setting_str).lower()
    if 'stereotype' in setting_str:
        return 'Stereotype'
    elif 'hair' in setting_str:
        return 'Hair Color'
    return None

df_res_t9['query_type'] = df_res_t9['setting'].apply(get_query_type)
df_res_t9 = df_res_t9[df_res_t9['query_type'].notna()]  # Drop anything that isn't Stereotype or Hair Color

# 4. Standardize Metric names to match LaTeX
df_res_t9['metric'] = df_res_t9['metric'].replace({
    'kl': 'KL',
    'max_skew': 'MS',
    'precision': 'Prec.'
})

# 5. Average over seeds and specific hair color settings
grouped = df_res_t9.groupby(['backbone', 'query_type', 'metric', 'method'], as_index=False)['mean'].mean()

# 6. Create the Pivot Table
pivot_df = grouped.pivot_table(
    index='method', 
    columns=['backbone', 'query_type', 'metric'], 
    values='mean'
)

# 7. Reorder Columns to perfectly match the LaTeX table
col_order_tuples = [
    ('RN50', 'Stereotype', 'KL'),
    ('RN50', 'Stereotype', 'MS'),
    ('RN50', 'Hair Color', 'KL'),
    ('RN50', 'Hair Color', 'MS'),
    ('RN50', 'Hair Color', 'Prec.'),
    ('RN101', 'Stereotype', 'KL'),
    ('RN101', 'Stereotype', 'MS'),
    ('RN101', 'Hair Color', 'KL'),
    ('RN101', 'Hair Color', 'MS'),
    ('RN101', 'Hair Color', 'Prec.')
]

col_order = pd.MultiIndex.from_tuples(
    col_order_tuples, 
    names=['Backbone', 'Query', 'Metric']
)

# Reindex and drop any columns that might be missing (prevents KeyErrors)
existing_cols = [col for col in col_order if col in pivot_df.columns]
pivot_df = pivot_df.reindex(columns=existing_cols)

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

print("\n" + "="*110)
print("TABLE 15 (ResNet): Measuring gender bias for stereotype and hair color queries on CelebA")
print("="*110)
print(pivot_df)
