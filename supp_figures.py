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
