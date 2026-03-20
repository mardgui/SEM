# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd
# import seaborn as sns
# import matplotlib.ticker as mticker

# # --- 1. Your Experimental Data ---
# # Order in this dict defines the column (facet) order
# results = {
#     'ViT-B/16': {
#         'Race': {
#             'chance': 1.0/7.0,
#             'CLIP': (0.9573, 0.006), 
#             'SAE': (0.7549, 0.0104)
#         },
#         'Gender': {
#             'chance': 0.50,
#             'CLIP': (0.9228, 0.0107),
#             'SAE': (0.7995, 0.0266)
#         }
#     },
#     'ViT-L/14@336px': {
#         'Race': {
#             'chance': 1.0/7.0,
#             'CLIP': (0.9491, 0.0071),
#             'SAE': (0.7104, 0.0148)  
#         },
#         'Gender': {
#             'chance': 0.50,
#             'CLIP': (0.8518, 0.042),
#             'SAE': (0.7475, 0.0327)
#         }
#     }
# }

# # --- 2. Helper Function for Normalization ---
# def normalize_score(acc, chance):
#     """
#     Normalizes the leakage accuracy.
#     Score 0.0 = Chance-level accuracy
#     Score 1.0 = Perfect (100%) accuracy
#     """
#     return (acc - chance) / (1.0 - chance)

# # --- 3. Data Preparation for Seaborn (Long Format) ---
# plot_data = []
# # These lists will define the plotting order
# ordered_backbones = list(results.keys())
# ordered_bias_types = list(results[ordered_backbones[0]].keys()) # Assumes same bias types for both
# ordered_methods = ['CLIP (Baseline)', 'CLIP+SAE (Ours)']

# for backbone, bias_data in results.items():
#     for bias_type, data in bias_data.items():
#         chance = data['chance']
        
#         # CLIP
#         clip_mean_norm = normalize_score(data['CLIP'][0], chance)
#         clip_std_norm = data['CLIP'][1] / (1.0 - chance) # Scale std dev
#         plot_data.append({
#             'Backbone': backbone,
#             'Bias Attribute': bias_type,
#             'Method': 'CLIP (Baseline)',
#             'Normalized Leakage': clip_mean_norm,
#             'Std Dev': clip_std_norm
#         })
        
#         # CLIP+SAE
#         sae_mean_norm = normalize_score(data['SAE'][0], chance)
#         sae_std_norm = data['SAE'][1] / (1.0 - chance) # Scale std dev
#         plot_data.append({
#             'Backbone': backbone,
#             'Bias Attribute': bias_type,
#             'Method': 'CLIP+SAE (Ours)',
#             'Normalized Leakage': sae_mean_norm,
#             'Std Dev': sae_std_norm
#         })

# df_plot = pd.DataFrame(plot_data)

# # Set the categorical orders in the DataFrame for correct plotting
# df_plot['Backbone'] = pd.Categorical(df_plot['Backbone'], categories=ordered_backbones, ordered=True)
# df_plot['Bias Attribute'] = pd.Categorical(df_plot['Bias Attribute'], categories=ordered_bias_types, ordered=True)
# df_plot['Method'] = pd.Categorical(df_plot['Method'], categories=ordered_methods, ordered=True)


# # --- 4. Plotting with Seaborn (FacetGrid + BarPlot) ---
# sns.set_theme(style="whitegrid")

# # Create a FacetGrid, with one column per 'Backbone'
# g = sns.catplot(
#     data=df_plot,
#     kind='bar',
#     x='Bias Attribute',     # X-axis is now the categories
#     y='Normalized Leakage', # Y-axis is now the value
#     hue='Method',
#     col='Backbone',         # <-- This creates the hierarchical grouping by column
#     col_order=ordered_backbones,
#     palette={'CLIP (Baseline)': 'C0', 'CLIP+SAE (Ours)': 'C1'},
#     errorbar=None,          # We will add error bars manually
#     height=4,               # Taller plot for vertical bars
#     aspect=1.2,             # Adjust aspect ratio
#     legend=False            # We will add a custom legend
# )

# # --- 5. Manually add Error Bars (Robust Method) ---
# # This new logic iterates by category, which is more robust
# n_methods = len(ordered_methods)
# n_bias_types = len(ordered_bias_types)

# for i, ax in enumerate(g.axes.flat):
#     backbone_name = ordered_backbones[i]
    
#     # Iterate over the x-axis categories (Race, Gender)
#     for j, bias_type in enumerate(ordered_bias_types):
        
#         # Iterate over the hues (CLIP, SAE)
#         for k, method in enumerate(ordered_methods):
            
#             # This is the patch for this specific (backbone, bias, method)
#             # The order in ax.patches is [CLIP-Race, SAE-Race, CLIP-Gender, SAE-Gender]
#             patch = ax.patches[j * n_methods + k]
            
#             # Get the corresponding data from the dataframe
#             data_row = df_plot[
#                 (df_plot['Backbone'] == backbone_name) &
#                 (df_plot['Bias Attribute'] == bias_type) &
#                 (df_plot['Method'] == method)
#             ]
            
#             std_dev = data_row['Std Dev'].iloc[0]
            
#             # Get coordinates from the bar
#             x_center = patch.get_x() + patch.get_width() / 2
#             y_top = patch.get_height()
            
#             # Draw the error bar
#             ax.errorbar(x_center, y_top, yerr=std_dev, fmt='none', color='black', capsize=4, elinewidth=1)

# # --- 6. Styling and Labels ---
# g.set_axis_labels('', 'Normalized Leakage (0 = Chance, 1 = Perfect)', fontsize=12)
# g.set_titles(col_template="{col_name}", size=12, weight='bold') # Set facet titles
# g.fig.suptitle('SAE Decomposition Significantly Reduces Information Leakage', fontsize=16, fontweight='bold', y=1.05)

# # Format Y-axis as percentage
# for ax in g.axes.flat:
#     ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
#     ax.set_ylim(bottom=0, top=1.05) # Give a little space at top
#     ax.grid(axis='y', linestyle=':', alpha=0.7)
#     ax.grid(axis='x', visible=False)

# # Add legend
# g.add_legend(title='Method', loc='upper left', bbox_to_anchor=(1.01, 1), frameon=True, fontsize=10)

# # Remove top and right spines
# sns.despine(left=False, bottom=True)

# # Adjust layout
# plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout to make space for legend

# # Save the figure
# output_filename = 'leakage_motivation_plot_vertical.pdf'
# plt.savefig(output_filename, bbox_inches='tight')

# print(f"Plot saved to '{output_filename}'")

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.ticker as mticker

# --- 1. Your Experimental Data ---
# Order in this dict defines the column (facet) order
results = {
    'ViT-B/16 (Race)': {
        'chance': 1.0/7.0,
        'CLIP': {'acc_b': (1.000, 0), 'acc_b_from_P': (0.9573, 0.006)},
        'SAE':  {'acc_b': (0.9995, 0.0004), 'acc_b_from_P': (0.7549, 0.0104)}
    },
    'ViT-B/16 (Gender)': {
        'chance': 0.50,
        'CLIP': {'acc_b': (0.9998, 0.0005), 'acc_b_from_P': (0.9228, 0.0107)},
        'SAE':  {'acc_b': (0.997, 0.002), 'acc_b_from_P': (0.7995, 0.0266)}
    },
    'ViT-L/14@336px (Race)': {
        'chance': 1.0/7.0,
        'CLIP': {'acc_b': (1.000, 0), 'acc_b_from_P': (0.9491, 0.0071)},
        'SAE':  {'acc_b': (0.9937, 0.0018), 'acc_b_from_P': (0.7104, 0.0148)} 
    },
    'ViT-L/14@336px (Gender)': {
        'chance': 0.50,
        'CLIP': {'acc_b': (1.000, 0), 'acc_b_from_P': (0.8518, 0.042)},
        'SAE':  {'acc_b': (0.9928, 0.004), 'acc_b_from_P': (0.7475, 0.0327)}
    }
}

# --- 2. Helper Function for Disentanglement Score ---
def calculate_disentanglement(acc_b_mean, acc_b_std, acc_b_from_P_mean, acc_b_from_P_std, chance):
    """
    Calculates the Disentanglement Score (D) and propagates error from both
    the numerator and the denominator.
    """
    # Calculate the mean values for the numerator (A) and denominator (B)
    actual_leakage_mean = acc_b_from_P_mean - chance
    max_leakage_mean = acc_b_mean - chance
    
    # Standard deviations for A and B (std(X - C) = std(X))
    actual_leakage_std = acc_b_from_P_std
    max_leakage_std = acc_b_std

    # Handle potential division by zero
    if max_leakage_mean <= 0 or actual_leakage_mean <= 0:
        return 0, 0 

    # --- Calculate Entanglement Ratio (E = A / B) ---
    entanglement_ratio_E = actual_leakage_mean / max_leakage_mean
    
    # --- Propagate Uncertainty for E = A / B ---
    # (sigma_E / E)^2 = (sigma_A / A)^2 + (sigma_B / B)^2
    relative_err_A_sq = (actual_leakage_std / actual_leakage_mean)**2
    relative_err_B_sq = (max_leakage_std / max_leakage_mean)**2
    
    std_dev_E = abs(entanglement_ratio_E) * np.sqrt(relative_err_A_sq + relative_err_B_sq)
    
    # --- Calculate Disentanglement Score (D = 1 - E) ---
    disentanglement_score_D = 1.0 - entanglement_ratio_E
    
    # The standard deviation of D is the same as the std dev of E
    # (since std(1 - X) = std(X))
    std_dev_D = std_dev_E
    
    return disentanglement_score_D, std_dev_D

# --- 3. Data Preparation for Seaborn (Long Format) ---
plot_data = []
ordered_backbones = ['ViT-B/16', 'ViT-L/14@336px']
ordered_bias_types = ['Race', 'Gender']
ordered_methods = ['CLIP (Baseline)', 'CLIP+SAE (Ours)']

for backbone in ordered_backbones:
    for bias_type in ordered_bias_types:
        data = results[f"{backbone} ({bias_type})"]
        chance = data['chance']
        
        # --- CLIP (Baseline) ---
        clip_data = data['CLIP']
        clip_acc_b_mean, clip_acc_b_std = clip_data['acc_b']
        clip_acc_b_from_P_mean, clip_acc_b_from_P_std = clip_data['acc_b_from_P']
        
        d_clip, d_clip_std = calculate_disentanglement(
            clip_acc_b_mean, clip_acc_b_std,
            clip_acc_b_from_P_mean, clip_acc_b_from_P_std,
            chance
        )
        plot_data.append({
            'Backbone': backbone,
            'Bias Attribute': bias_type,
            'Method': 'CLIP (Baseline)',
            'Disentanglement Score': d_clip,
            'Std Dev': d_clip_std
        })
        
        # --- CLIP+SAE (Ours) ---
        sae_data = data['SAE']
        sae_acc_b_mean, sae_acc_b_std = sae_data['acc_b']
        sae_acc_b_from_P_mean, sae_acc_b_from_P_std = sae_data['acc_b_from_P']

        d_sae, d_sae_std = calculate_disentanglement(
            sae_acc_b_mean, sae_acc_b_std,
            sae_acc_b_from_P_mean, sae_acc_b_from_P_std,
            chance
        )
        plot_data.append({
            'Backbone': backbone,
            'Bias Attribute': bias_type,
            'Method': 'CLIP+SAE (Ours)',
            'Disentanglement Score': d_sae,
            'Std Dev': d_sae_std
        })

df_plot = pd.DataFrame(plot_data)

# Set the categorical orders in the DataFrame for correct plotting
df_plot['Backbone'] = pd.Categorical(df_plot['Backbone'], categories=ordered_backbones, ordered=True)
df_plot['Bias Attribute'] = pd.Categorical(df_plot['Bias Attribute'], categories=ordered_bias_types, ordered=True)
df_plot['Method'] = pd.Categorical(df_plot['Method'], categories=ordered_methods, ordered=True)


# # --- 4. Plotting with Seaborn (FacetGrid + BarPlot) ---
# sns.set_theme(style="whitegrid")

# # Create a FacetGrid, with one column per 'Backbone'
# g = sns.catplot(
#     data=df_plot,
#     kind='bar',
#     x='Bias Attribute',     # X-axis is now the categories
#     y='Disentanglement Score', # Y-axis is now the value
#     hue='Method',
#     col='Backbone',         # <-- This creates the hierarchical grouping by column
#     col_order=ordered_backbones,
#     palette={'CLIP (Baseline)': 'C0', 'CLIP+SAE (Ours)': 'C1'},
#     errorbar=None,          # We will add error bars manually
#     height=4,               # <-- CHANGED: Made plot taller
#     aspect=0.6,             # <-- CHANGED: Made plot narrower
#     legend=True             # <-- CHANGED: Let seaborn create the legend
# )

# # --- 5. Manually add Error Bars (Robust Method) ---
# # This new logic iterates by category, which is more robust
# n_methods = len(ordered_methods)
# n_bias_types = len(ordered_bias_types)

# for i, ax in enumerate(g.axes.flat):
#     backbone_name = ordered_backbones[i]
    
#     # Iterate over the x-axis categories (Race, Gender)
#     for j, bias_type in enumerate(ordered_bias_types):
        
#         # Iterate over the hues (CLIP, SAE)
#         for k, method in enumerate(ordered_methods):
            
#             # This is the patch for this specific (backbone, bias, method)
#             # The order in ax.patches is [CLIP-Race, SAE-Race, CLIP-Gender, SAE-Gender]
#             patch = ax.patches[j * n_methods + k]
            
#             # Get the corresponding data from the dataframe
#             data_row = df_plot[
#                 (df_plot['Backbone'] == backbone_name) &
#                 (df_plot['Bias Attribute'] == bias_type) &
#                 (df_plot['Method'] == method)
#             ]
            
#             std_dev = data_row['Std Dev'].iloc[0]
            
#             # Get coordinates from the bar
#             x_center = patch.get_x() + patch.get_width() / 2
#             y_top = patch.get_height()
            
#             # Draw the error bar
#             ax.errorbar(x_center, y_top, yerr=std_dev, fmt='none', color='black', capsize=4, elinewidth=1)

# # --- 6. Styling and Labels ---
# # g.set_axis_labels('', 'Normalized Leakage (0 = Chance, 1 = Perfect)', fontsize=12)
# g.set_axis_labels('', 'Disentanglement Score', fontsize=18)
# g.set_titles(col_template="{col_name}", size=16, weight='bold') # Set facet titles
# # g.fig.suptitle('SAEs Create a More Disentangled Feature Space', fontsize=16, fontweight='bold', y=1.05)

# # Format Y-axis as percentage
# for ax in g.axes.flat:
#     ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
#     ax.set_ylim(bottom=0, top=0.65) # Give a little space at top
#     ax.tick_params(axis='x', labelsize=14)
#     ax.tick_params(axis='y', labelsize=14)
#     ax.grid(axis='y', linestyle=':', alpha=0.7)
#     ax.grid(axis='x', visible=False)

# # --- CHANGED: Fix for Legend ---
# # Move the legend that seaborn automatically created
# sns.move_legend(
#     g, "upper left", 
#     bbox_to_anchor=(.14, .9), 
#     title='Method', 
#     frameon=True,
#     fontsize=13,      # <-- Add this line for the item font size
#     title_fontsize=14 # <-- Add this line for the title font size
# )
# # --- End of Change ---

# # Remove top and right spines
# sns.despine(left=False, bottom=True)

# # Adjust layout
# # plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout to make space for legend
# plt.tight_layout()

# # Save the figure
# output_filename = 'disentanglement_score_plot.pdf'
# plt.savefig(output_filename, bbox_inches='tight')

# print(f"Plot saved to '{output_filename}'")

# --- 4. Plotting with Seaborn (FacetGrid + BarPlot) ---
sns.set_theme(style="whitegrid")

# --- ADDED: Set global font to serif for consistency ---
matplotlib.rcParams.update({
    'font.family': 'serif',
    'mathtext.fontset': 'stix' # Use STIX serif font for math
})

# --- ADDED: Define consistent color palette ---
colors = sns.color_palette("colorblind", 2)
custom_palette = {'CLIP (Baseline)': colors[0], 'CLIP+SAE (Ours)': colors[1]}

# Create a FacetGrid, with one column per 'Backbone'
g = sns.catplot(
    data=df_plot,
    kind='bar',
    x='Bias Attribute',
    y='Disentanglement Score',
    hue='Method',
    col='Backbone',
    col_order=ordered_backbones,
    palette=custom_palette, # <-- UPDATED to use consistent palette
    errorbar=None,
    height=4, 
    aspect=0.7,
    legend=True
)

# --- 5. Manually add Error Bars (Robust Method) ---
# ... (Error bar loop is unchanged) ...
n_methods = len(ordered_methods)
n_bias_types = len(ordered_bias_types)

for i, ax in enumerate(g.axes.flat):
    backbone_name = ordered_backbones[i]
    
    for j, bias_type in enumerate(ordered_bias_types):
        for k, method in enumerate(ordered_methods):
            
            patch = ax.patches[j * n_methods + k]
            
            data_row = df_plot[
                (df_plot['Backbone'] == backbone_name) &
                (df_plot['Bias Attribute'] == bias_type) &
                (df_plot['Method'] == method)
            ]
            
            std_dev = data_row['Std Dev'].iloc[0]
            y_top = patch.get_height()
            x_center = patch.get_x() + patch.get_width() / 2
            
            ax.errorbar(x_center, y_top, yerr=std_dev, fmt='none', color='black', capsize=4, elinewidth=1)

# --- 6. Styling and Labels ---
# ... (Styling code is unchanged, but will now use the serif font) ...
g.set_axis_labels('', 'Disentanglement Score', fontsize=18)
g.set_titles(col_template="{col_name}", size=16, weight='bold') 
# g.fig.suptitle('SAEs Create a More Disentangled Feature Space', fontsize=16, fontweight='bold', y=1.05)

for ax in g.axes.flat:
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax.set_ylim(bottom=0, top=0.65)
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)
    ax.grid(axis='y', linestyle=':', alpha=0.7)
    ax.grid(axis='x', visible=False)

sns.move_legend(
    g, "upper left", 
    bbox_to_anchor=(.14, .9), 
    title='Method', 
    frameon=True,
    fontsize=13,
    title_fontsize=14
)

sns.despine(left=False, bottom=True)
plt.tight_layout()

# Save the figure
output_filename = 'disentanglement_score_plot.pdf'
plt.savefig(output_filename, bbox_inches='tight')

print(f"Plot saved to '{output_filename}'")