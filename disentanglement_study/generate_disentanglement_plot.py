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

# --- 4. Plotting with Seaborn (FacetGrid + BarPlot) ---
sns.set_theme(style="whitegrid")

# --- Set global font to serif for consistency ---
matplotlib.rcParams.update({
    'font.family': 'serif',
    'mathtext.fontset': 'stix' # Use STIX serif font for math
})

# --- Define consistent color palette ---
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
    palette=custom_palette,
    errorbar=None,
    height=4, 
    aspect=0.7,
    legend=True
)

# --- 5. Manually add Error Bars ---
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
g.set_axis_labels('', 'Disentanglement Score', fontsize=18)
g.set_titles(col_template="{col_name}", size=16, weight='bold') 

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
