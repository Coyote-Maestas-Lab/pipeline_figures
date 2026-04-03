import os

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch

# fix matplotlib font issue
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['pdf.fonttype'] = 42
# plt.rcParams['ps.fonttype'] = 42
# plt.rcParams['svg.fonttype'] = 'none'


def create_chimerax_file(output_df: pd.DataFrame, output_dir: str, attribute_name: str,
                         descriptive_text: str = "") -> None:
    """
    Create a ChimeraX .defattr file for visualizing annotations.

    Parameters
    ----------
    output_df : pd.DataFrame
        DataFrame containing 'chain', 'resi', and attribute_name columns.
    output_dir : str
        Directory to save the .defattr file.
    attribute_name : str
        Name of the attribute to visualize.
    descriptive_text : str
        Descriptive text for the attribute file (default is empty).

    Returns
    -------
    None
    """

    # headers for .defattr file
    key_values = {
        "attribute": attribute_name,
        "match mode": "1-to-1",
        "recipient": "residues"
    }

    with open(os.path.join(output_dir, f"{attribute_name}.defattr"), "w") as f:
        # write descriptive text if provided
        if descriptive_text:
            f.write(f"# {descriptive_text}\n")

        # write headers
        for k, v in key_values.items():
                f.write(f"{k}: {v}\n")

        # write attributes
        for _, row in output_df.iterrows():
            f.write(f"\t/{row.chain}:{int(row.resi_struct)}\t{row[attribute_name]}\n")

plot_dir = '/Users/ngreenwald/Library/CloudStorage/Box-Box/WCM Lab/Noah/feature_pipeline/fig3'

scores_abund = pd.read_csv(f'{plot_dir}/8ET6_OCT1_features.csv')
metadata = pd.read_csv(f'{plot_dir}/8ET6_OCT1_metadata.csv')
bonds = pd.read_csv(f'{plot_dir}/8ET6_OCT1_bonds.csv')
scores_surv = pd.read_csv(f'{plot_dir}/8ET6_OCT1_survival_features.csv')

# Create per-position df
position_scores = scores_abund.drop_duplicates(subset=['resi_mut', 'resn_mut'])

# simplify secondary structure domains to higher level categories
position_scores['ss_group_simple'] = 'TMD'
position_scores.loc[position_scores.ss_domains.str.contains('loop'), 'ss_group_simple'] = 'loop'
position_scores.loc[position_scores.ss_domains.str.contains('sheet'), 'ss_group_simple'] = 'sheet'

# --- Heatmap + metadata strips (self-contained) ---
mutation_order = [
    "ALA", "ILE", "LEU", "VAL", "MET", "PHE", "TYR", "TRP", "SER", "THR",
    "ASN", "GLN", "HIS", "LYS", "ARG", "ASP", "GLU", "GLY", "PRO", "CYS", "DEL0"
]
meta_cols = ["effect_quartile"]
heatmap_cmap_name = "bwr_r"
heatmap_vcenter = 0
heatmap_vmin = -2
heatmap_vmax = 0.75
effect_quartile_order = ['Q1', 'Q2', 'Q3', 'Q4']
heatmap_cmap = plt.get_cmap(heatmap_cmap_name)
heatmap_norm = mcolors.TwoSlopeNorm(vmin=heatmap_vmin, vcenter=heatmap_vcenter, vmax=heatmap_vmax)

quartile_sample_values = np.linspace(
    heatmap_vmin,
    heatmap_vmax,
    len(effect_quartile_order),
)
effect_quartile_palette = {
    quartile: heatmap_cmap(heatmap_norm(sample_value))
    for quartile, sample_value in zip(effect_quartile_order, quartile_sample_values)
}

# Effect heatmap (rows are mutation types, columns are positions).
heatmap_df_effect = (
    scores_abund.pivot(index="resm", columns="resi_mut", values="effect")
    .reindex(index=mutation_order)
)

# One metadata value per `resi_mut` column (aligned to the heatmap columns).
pos_meta = (
    position_scores.drop_duplicates(subset=["resi_mut"])[["resi_mut"] + meta_cols]
    .set_index("resi_mut")
    .reindex(heatmap_df_effect.columns)
)

# Convert the single metadata field into a color strip.
category_to_color = {"effect_quartile": effect_quartile_palette}
col_colors = pd.DataFrame(
    {"effect_quartile": pos_meta["effect_quartile"].astype(str).map(effect_quartile_palette)},
    index=pos_meta.index,
)

cg = sns.clustermap(
    heatmap_df_effect,
    row_cluster=False,
    col_cluster=False,
    col_colors=col_colors,
    cmap=heatmap_cmap_name,
    center=heatmap_vcenter,
    figsize=(20, 4),
    vmin=heatmap_vmin,
    vmax=heatmap_vmax
)
position_labels = heatmap_df_effect.columns.to_numpy()
xtick_idx = np.where(position_labels % 20 == 0)[0]
cg.ax_heatmap.set_xticks(xtick_idx + 0.5)
cg.ax_heatmap.set_xticklabels(position_labels[xtick_idx].astype(int).astype(str), rotation=0)
cg.ax_heatmap.tick_params(axis='x', labelsize=11, rotation=0)
cg.ax_heatmap.set_yticks(np.arange(len(heatmap_df_effect.index)) + 0.5)
cg.ax_heatmap.set_yticklabels(heatmap_df_effect.index.astype(str), rotation=0)
cg.ax_heatmap.tick_params(axis='y', labelsize=7)

# Legend (field + category).
handles, labels = [], []
for field in meta_cols:
    for cat, color in category_to_color[field].items():
        handles.append(Patch(facecolor=color, edgecolor="none"))
        labels.append(f"{field}: {cat}")
cg.figure.legend(handles, labels, loc="lower left", bbox_to_anchor=(0.02, 0.02), frameon=False, fontsize=8)

cg.savefig(f"{plot_dir}/fig3a_abundance_heatmap_with_metadata.pdf", bbox_inches="tight")
plt.close(cg.figure)


position_scores = scores_abund.drop_duplicates(subset=['resi_mut'])
quartile_positions = position_scores['pos_effect'].quantile([0.25, 0.5, 0.75])

# Make a single KDE plot for pos_effect
fig, ax = plt.subplots(figsize=(20, 10))
sns.kdeplot(x='pos_effect', data=position_scores, ax=ax, fill=True)
# draw vertical lines at the quartile positions
for quartile in quartile_positions:
    ax.axvline(x=quartile, color='black', linestyle='--')
# remove upper right hand corner of plot
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.savefig(f'{plot_dir}/fig3b_pos_effect_kde.pdf')
plt.close()

# Show location of Q1, Q2, Q3, Q4 positions on the structure for 3c
output_df = scores_abund.drop_duplicates(subset=['chain', 'resi_struct', 'resn_struct'])
output_df = output_df[['chain', 'resi_struct', 'resn_struct', 'effect_quartile']]
output_df = output_df.loc[output_df.resi_struct.notna(), :]
numeric_dict = {'Q1': 4, 'Q2': 3, 'Q3': 2, 'Q4': 1}
output_df['effect_quartile'] = output_df['effect_quartile'].map(numeric_dict).fillna(5)

create_chimerax_file(output_df=output_df,
                    output_dir=f'{plot_dir}',
                    attribute_name='effect_quartile',
                    descriptive_text='Numeric Effect Quartile')

# Plot boxplot showing abundance effect for positions starting out as negatively charged
fig, ax = plt.subplots(figsize=(10, 5))
sns.boxplot(x='mut_aa_group', y='effect', hue='effect_quartile',data=scores_abund.loc[scores_abund.wildtype_aa_group == 'Negatively_Charged', :], ax=ax,
            hue_order=effect_quartile_order, palette=effect_quartile_palette,
            order=['Aromatic', 'Nonpolar_Aliphatic', 'Special', 'Positively_Charged', 'Polar_Uncharged',  'Negatively_Charged'])
plt.savefig(f'{plot_dir}/fig3d_abundance_effect_for_negatively_charged_positions_boxplot.pdf')
plt.close()



# Plot distance to surface residue for Q1, Q2, Q3, Q4 positions
fig, ax = plt.subplots(figsize=(6, 3))
sns.violinplot(x='effect_quartile', y='distance_to_nearest_surface_residue', data=position_scores, ax=ax, order=['Q1', 'Q2', 'Q3', 'Q4'],
hue='effect_quartile', hue_order=effect_quartile_order, palette=effect_quartile_palette)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.savefig(f'{plot_dir}/fig3e_distance_to_surface_residue_for_quartile_positions_violinplot.pdf')
plt.close()

# Plot total bond count across quartiles
fig, ax = plt.subplots(figsize=(6, 3))
sns.violinplot(x='effect_quartile', y='total_bond_count', data=position_scores, ax=ax, order=['Q1', 'Q2', 'Q3', 'Q4'],
hue='effect_quartile', hue_order=effect_quartile_order, palette=effect_quartile_palette)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.savefig(f'{plot_dir}/fig3f_total_bond_count_for_quartile_positions_violinplot.pdf')
plt.close()

# scatterplot of blosum score vs effect
fig, ax = plt.subplots(figsize=(6, 3))
sns.boxplot(x='blosum90', y='effect', data=scores_abund, ax=ax)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.savefig(f'{plot_dir}/fig3g_blosum90_vs_effect_boxplot.pdf')
plt.close()
