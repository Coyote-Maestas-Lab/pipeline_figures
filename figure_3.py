import os

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt



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

plot_dir = '/Users/ngreenwald/Library/CloudStorage/Box-Box/WCM Lab/Noah/feature_pipeline/20260326_fig3'

scores_abund = pd.read_csv(f'{plot_dir}/8ET6_OCT1_features.csv')
metadata = pd.read_csv(f'{plot_dir}/8ET6_OCT1_metadata.csv')
bonds = pd.read_csv(f'{plot_dir}/8ET6_OCT1_bonds.csv')
scores_surv = pd.read_csv(f'{plot_dir}/8ET6_OCT1_survival_features.csv')


# create heatmap of abundance by mutation at each position
mutation_order = ['ALA', 'ILE', 'LEU', 'VAL', 'MET', 'PHE', 'TYR', 'TRP', 'SER', 'THR', 'ASN', 'GLN', 'HIS', 'LYS', 'ARG', 'ASP', 'GLU', 'GLY', 'PRO', 'CYS', 'DEL0']

fig, ax = plt.subplots(figsize=(20, 5))
heatmap_df = scores_abund.pivot(index='resm', columns='resi_mut', values='effect')
heatmap_df = heatmap_df.reindex(index=mutation_order)
sns.heatmap(heatmap_df, ax=ax, cmap='RdBu_r', center=0)
ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
position_labels = heatmap_df.columns.to_numpy()
xtick_idx = np.where(position_labels % 20 == 0)[0]
ax.set_xticks(xtick_idx + 0.5)
ax.set_xticklabels(position_labels[xtick_idx].astype(int).astype(str), rotation=0)
ax.tick_params(axis='x', labelsize=11, rotation=0)
ax.tick_params(axis='y', labelsize=10)
plt.title('Abundance Effect by Mutation at Each Position')
plt.savefig(f'{plot_dir}/abundance_heatmap.pdf')
plt.close()


# Plot boxplot showing distance to nearest surface residue for each effect quartile
fig, ax = plt.subplots(figsize=(20, 10))
order = ['Q1', 'Q2', 'Q3', 'Q4']
sns.boxplot(x='effect_quartile', y='distance_to_nearest_surface_residue', data=scores, ax=ax, order=order)
plt.savefig(f'{plot_dir}/distance_to_nearest_surface_residue_boxplot.png')
plt.close()

# plot boxplot showing abundance effect for positions starting out as aromatic
fig, ax = plt.subplots(figsize=(20, 10))
sns.boxplot(x='mut_aa_group', y='effect', hue='effect_quartile',data=scores_abund.loc[scores_abund.wildtype_aa_group == 'Aromatic', :], ax=ax, hue_order=order)
plt.savefig(f'{plot_dir}/abundance_effect_for_aromatic_positions_boxplot.png')
plt.close()

# Plot boxplot showing abundance effect for positions starting out as negatively charged
fig, ax = plt.subplots(figsize=(20, 10))
sns.boxplot(x='mut_aa_group', y='effect', hue='effect_quartile',data=scores_abund.loc[scores_abund.wildtype_aa_group == 'Negatively_Charged', :], ax=ax, hue_order=order)
plt.savefig(f'{plot_dir}/abundance_effect_for_negatively_charged_positions_boxplot.png')
plt.close()

# plot scaatterplot of abundance effect vs SASA in a 2x2 grid with each effect quartile in a subplot
fig, ax = plt.subplots(figsize=(20, 10), ncols=2, nrows=2)
for i, ax_ in enumerate(ax.flatten()):
    order = ['Q1', 'Q2', 'Q3', 'Q4']
    ax_df = scores_abund.loc[scores_abund.effect_quartile == order[i], :]
    sns.scatterplot(x='sasa', y='effect', data=ax_df, ax=ax_)
    ax_.set_title(f'Effect Quartile {order[i]}')
plt.savefig(f'{plot_dir}/abundance_effect_vs_sasa_scatterplot.png')
plt.close()
