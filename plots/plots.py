# Evaluate generate metrics for desingated proteins
# from src.pipeline.runner import Runner
import pandas as pd

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



input_dir = '/Users/ngreenwald/Library/CloudStorage/Box-Box/WCM Lab/Noah/biogenesis'
working_dir = '/Users/ngreenwald/Library/CloudStorage/Box-Box/WCM Lab/Noah/feature_pipeline/20260309_output'

ids = pd.read_csv(f'{input_dir}/inputs/pdb_ids.csv')
keep_subset = ['Kir2.1', 'OCT_1', 'GPR68']
ids = ids[ids['name'].isin(keep_subset)]

for _, row in ids.iterrows():
    pdb_id = row['pdb_id']
    protein = row['name']
    chain = row['chain']
    membrane_protein = row['membrane_protein']
    config_path = f'{input_dir}/inputs/chain{chain}_config.toml'
    runner = Runner(
        name=f'{protein}_test',
        pdb_id=pdb_id,
        membrane_protein=membrane_protein,
        mutation_data_path=f'{input_dir}/inputs/{protein}_formatted_scores.csv',
        config_path=config_path
    )
    runner.run()
    runner.save_results(output_dir=f'{working_dir}')

scores = pd.read_csv(f'{working_dir}/9BI6_features.csv')
scores = scores.loc[scores.chain == 'R', :]

plot_dir = '/Users/ngreenwald/Library/CloudStorage/Box-Box/WCM Lab/Noah/feature_pipeline/20260310_fig2'


# Assign features to buckets based on type
all_features = scores.columns.tolist()

def assign_features(feature_names: list[str], exact_names: set[str], prefixes: tuple[str, ...] = ()) -> list[str]:
    return [
        feature
        for feature in feature_names
        if feature in exact_names or feature.startswith(prefixes)
    ]


# Features that require mutation data to calculate
mut_features = assign_features(
    all_features,
    exact_names={
        'effect',
        'pos_effect',
        'effect_quartile',
        'effect_variance',
        'effect_variance_rank',
        'effect_ranking',
        'blosum90',
        'phat_score',
        'wildtype_aa_group',
        'mut_aa_group',
        'wildtype_mut_aa_group',
    },
    prefixes=('kidera_', 'AAIndex_'),
)

# Features that summarize information about a single residue
single_residue_features = assign_features(
    all_features,
    exact_names={
        'sasa',
        'sasa_backbone',
        'sasa_sidechain',
        'sasa_polar',
        'sasa_nonpolar',
        'kyte_doolittle',
    },
)

# Features that summarize information about interactions between residues
interaction_features = assign_features(
    all_features,
    exact_names={
        'salt_bridge_count',
        'ionic_bond_count',
        'disulfide_bond_count',
        'pi_stacking_count',
        'cation_pi_count',
        'vdw_contact_count',
        'bb_hbond_count',
        'sc_hbond_count',
        'total_hbond_count',
        'total_bond_count',
        'total_within_chain_bonds',
        'total_between_chain_bonds',
    },
    prefixes=(),
)

# Features that summarize information about secondary or tertiary structure
ss_features = assign_features(
    all_features,
    exact_names={
        'ss_group',
        'ss_domains',
        'distance_to_nearest_surface_residue',
        'distance_from_membrane_edge',
        'distance_to_center_of_mass',
        'ss_domain_length',
         'packing_n_atoms',
        'packing_n_neighbor_residues',
        'packing_contact_density',
        'n_ala_neighbors'
    },
    prefixes=('dssp_', 'ss_domain_', 'neighborhood_', 'graph_all_', 'graph_vdw_contact_', 'graph_hbond_'),
)

# Features that summarize information about ligand interactions
ligand_features = assign_features(
    all_features,
    exact_names=set(),
    prefixes=('ligand_',),
)

assigned_features = (
    set(mut_features)
    | set(single_residue_features)
    | set(interaction_features)
    | set(ss_features)
    | set(ligand_features)
)
undetermined_features = [f for f in all_features if f not in assigned_features]

# plot percentages of features by type
feature_types = ['mut_features', 'single_residue_features', 'interaction_features', 'ss_features', 'ligand_features']
feature_type_counts = [len(mut_features), len(single_residue_features), len(interaction_features), len(ss_features), len(ligand_features)]
feature_type_percentages = [count / len(all_features) for count in feature_type_counts]
fig, ax = plt.subplots(figsize=(20, 10))
sns.barplot(x=feature_types, y=feature_type_percentages, ax=ax)
plt.title('Percentages of Features by Type')
plt.savefig(f'{plot_dir}/percentages_of_features_by_type_barplot.png')
plt.close()


# Figure 3

position_scores = scores.drop_duplicates(subset=['resi_mut', 'resn_mut'])
# Generate stacked bar showing proportion of residues in each wildtype_aa_group by effect quartile
wildtype_aa_group_proportions = position_scores.groupby('effect_quartile')['wildtype_aa_group'].value_counts(normalize=True).reset_index(name='proportion')
fig, ax = plt.subplots(figsize=(20, 10))
order = ['Q1', 'Q2', 'Q3', 'Q4']
sns.barplot(x='effect_quartile', y='proportion', data=wildtype_aa_group_proportions, ax=ax, order=order, hue='wildtype_aa_group')
plt.title('Proportion of residues in each wildtype_aa_group by effect quartile')
plt.savefig(f'{plot_dir}/wildtype_aa_group_by_effect_quartile_stacked_bar.png')
plt.close()


# Show location of Q1, Q2, Q3, Q4 positions on the structure
output_df = scores.drop_duplicates(subset=['chain', 'resi_struct', 'resn_struct'])
output_df = output_df[['chain', 'resi_struct', 'resn_struct', 'effect_quartile']]
output_df = output_df.loc[output_df.resi_struct.notna(), :]
numeric_dict = {'Q1': 4, 'Q2': 3, 'Q3': 2, 'Q4': 1}
output_df['effect_quartile'] = output_df['effect_quartile'].map(numeric_dict).fillna(5)

create_chimerax_file(output_df=output_df,
                    output_dir=f'{plot_dir}',
                    attribute_name='effect_quartile',
                    descriptive_text='Numeric Effect Quartile')

# Show location of effect scores on the structure
output_df = scores.drop_duplicates(subset=['chain', 'resi_struct', 'resn_struct'])
output_df = output_df[['chain', 'resi_struct', 'resn_struct', 'effect']]
output_df = output_df.loc[output_df.resi_struct.notna(), :]
create_chimerax_file(output_df=output_df,
                    output_dir=f'{plot_dir}',
                    attribute_name='effect',
                    descriptive_text='Effect')


# Show location of variable positions on the structure
output_df = scores.drop_duplicates(subset=['chain', 'resi_struct', 'resn_struct'])
output_df = output_df[['chain', 'resi_struct', 'resn_struct', 'effect_variance_rank']]
output_df = output_df.loc[output_df.resi_struct.notna(), :]
create_chimerax_file(output_df=output_df,
                    output_dir=f'{plot_dir}',
                    attribute_name='effect_variance_rank',
                    descriptive_text='Effect Variance Rank')

# Scatter of distance to membrane edge vs effect
fig, ax = plt.subplots(figsize=(20, 10))
sns.scatterplot(x='distance_from_membrane_edge', y='effect', data=scores, ax=ax)
plt.title('Distance to Membrane Edge vs Effect')
plt.savefig(f'{plot_dir}/distance_from_membrane_edge_vs_effect_scatterplot.png')
plt.close()

# kidera factors
fig, ax = plt.subplots(figsize=(20, 10))
sns.scatterplot(x='kidera_f1_wt', y='effect', data=scores, ax=ax)
plt.title('Kidera Factor 1 vs Effect')
plt.savefig(f'{plot_dir}/kidera_f1_wt_vs_effect_scatterplot.png')
plt.close()

# kidera factors diff
kidera_factors = ['kidera_f1_diff', 'kidera_f2_diff', 'kidera_f3_diff', 'kidera_f4_diff', 'kidera_f5_diff', 'kidera_f6_diff', 'kidera_f7_diff', 'kidera_f8_diff', 'kidera_f9_diff', 'kidera_f10_diff']

for kidera_factor in kidera_factors:
    fig, ax = plt.subplots(1, 4, figsize=(30, 10), sharex=True, sharey=True)
    for i, ax_ in enumerate(ax.flatten()):
        order = ['Q1', 'Q2', 'Q3', 'Q4']
        ax_df = scores.loc[scores.effect_quartile == order[i], :]
        sns.kdeplot(x=kidera_factor, y='effect', data=ax_df, ax=ax_, alpha=0.5)
        ax_.set_title(f'Effect Quartile {order[i]}')
    fig.suptitle(f'{kidera_factor}')
    plt.savefig(f'{plot_dir}/{kidera_factor}_vs_effect_quartile_kdeplot.png')
    plt.close()


# kidera factors mut
kidera_factors_mut = ['kidera_f1_mut', 'kidera_f2_mut', 'kidera_f3_mut', 'kidera_f4_mut', 'kidera_f5_mut', 'kidera_f6_mut', 'kidera_f7_mut', 'kidera_f8_mut', 'kidera_f9_mut', 'kidera_f10_mut']
for kidera_factor_mut in kidera_factors_mut:
    fig, ax = plt.subplots(1, 4, figsize=(30, 10), sharex=True, sharey=True)
    for i, ax_ in enumerate(ax.flatten()):
        order = ['Q1', 'Q2', 'Q3', 'Q4']
        ax_df = scores.loc[scores.effect_quartile == order[i], :]
        sns.kdeplot(x=kidera_factor_mut, y='effect', data=ax_df, ax=ax_, alpha=0.5)
        ax_.set_title(f'Effect Quartile {order[i]}')
    fig.suptitle(f'{kidera_factor_mut}')
    plt.savefig(f'{plot_dir}/{kidera_factor_mut}_vs_effect_quartile_kdeplot.png')
    plt.close()

# plot position effect vs variance
fig, ax = plt.subplots(figsize=(20, 10))
sns.scatterplot(x='effect_variance', y='pos_effect', data=position_scores, ax=ax, hue='effect_quartile')
plt.title('Position Effect vs Variance')
plt.savefig(f'{plot_dir}/pos_effect_vs_variance_scatterplot.png')
plt.close()

# Generate a heatmap of effects by position by mutation, sorted by effect variance
fig, ax = plt.subplots(figsize=(20, 10))
ranked_scores = position_scores.sort_values(by="effect_variance", ascending=False)
resi_mut_order = ranked_scores["resi_mut"].drop_duplicates()

heatmap_df = scores.pivot(index="resm", columns="resi_mut", values="effect")
heatmap_df = heatmap_df.reindex(columns=resi_mut_order)  # enforce column order

sns.heatmap(heatmap_df, ax=ax, cmap='RdBu_r')
plt.title('Effects by Position by Mutation, Sorted by Effect Variance')
plt.savefig(f'{plot_dir}/effects_by_position_by_mutation_sorted_by_variance_heatmap.png')
plt.close()

# identify low variance, Q1 positions, annotate position_scores
position_scores['variance_cat'] = 'other'
position_scores.loc[np.logical_and(position_scores.effect_variance_rank < 0.25, position_scores.effect_quartile == 'Q1'), 'variance_cat'] = 'low_variance_Q1'
position_scores.loc[np.logical_and(position_scores.effect_variance_rank >0.5, position_scores.effect_quartile == 'Q1'), 'variance_cat'] = 'high_variance_Q1'

mapping = {'other': 0, 'low_variance_Q1': 1, 'high_variance_Q1': 2}
position_scores['variance_cat'] = position_scores['variance_cat'].map(mapping)

input = position_scores[['chain', 'resi_struct', 'resn_struct', 'variance_cat']]
input = input.loc[input.resi_struct.notna(), :]
create_chimerax_file(output_df=input,
                    output_dir=f'{plot_dir}',
                    attribute_name='variance_cat',
                    descriptive_text='Variance Category')


# summarize counts of variance cat by wildtype and mut aa group
variance_cat_counts = position_scores.loc[position_scores.variance_cat != 'other', :].groupby(['wildtype_aa_group', 'variance_cat']).size().reset_index(name='count')
fig, ax = plt.subplots(figsize=(20, 10))
sns.barplot(x='wildtype_aa_group', y='count', data=variance_cat_counts, ax=ax, hue='variance_cat')
plt.title('Variance Category by Wildtype and Mut AA Group')
plt.savefig(f'{plot_dir}/variance_cat_by_wildtype_aa_group_barplot.png')
plt.close()

# summarize counts of variance cat by effect quartile
variance_cat_counts = position_scores.loc[position_scores.variance_cat != 'other', :].groupby(['effect_quartile', 'variance_cat']).size().reset_index(name='count')
fig, ax = plt.subplots(figsize=(20, 10))
sns.barplot(x='effect_quartile', y='count', data=variance_cat_counts, ax=ax, hue='variance_cat')
plt.title('Variance Category by Effect Quartile')
plt.savefig(f'{plot_dir}/variance_cat_by_effect_quartile_barplot.png')
plt.close()
