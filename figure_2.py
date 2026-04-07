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

plot_dir = '/Users/ngreenwald/Library/CloudStorage/Box-Box/WCM Lab/Noah/feature_pipeline/fig2'


scores = pd.read_csv(f'{plot_dir}/5C1M_MOR_features.csv')
metadata = pd.read_csv(f'{plot_dir}/5C1M_MOR_metadata.csv')

# Generate chimerax file for SASA
output_df = scores.drop_duplicates(subset=['chain', 'resi_struct', 'resn_struct'])
output_df = output_df[['chain', 'resi_struct', 'resn_struct', 'sasa']]
output_df = output_df.loc[output_df.resi_struct.notna(), :]
create_chimerax_file(output_df=output_df,
                    output_dir=f'{plot_dir}',
                    attribute_name='sasa',
                    descriptive_text='SASA')

# generate chimerax file for secondary structure
output_df = metadata.drop_duplicates(subset=['chain', 'resi_struct', 'resn_struct'])
output_df = output_df[['chain', 'resi_struct', 'resn_struct', 'ss_group']]
output_df = output_df.loc[output_df.resi_struct.notna(), :]
output_df.loc[output_df.ss_group.str.contains('c'), 'ss_group'] = 'c'
output_df.loc[output_df.ss_group.str.contains('b'), 'ss_group'] = 'b'
output_df.loc[output_df.ss_group.str.contains('a'), 'ss_group'] = 'a'
mapping = {'a': 1, 'b': 2, 'c': 3}
output_df['ss_group'] = output_df['ss_group'].map(mapping)
create_chimerax_file(output_df=output_df,
                    output_dir=f'{plot_dir}',
                    attribute_name='ss_group',
                    descriptive_text='Secondary Structure')


# Generate chimerax file for ligand_A_1401_P0G_interactions
output_df = scores.drop_duplicates(subset=['chain', 'resi_struct', 'resn_struct'])
output_df = output_df[['chain', 'resi_struct', 'resn_struct', 'ligand_A_407_VF1_interactions']]
output_df = output_df.loc[output_df.resi_struct.notna(), :]
output_df['ligand_numeric'] = 0
output_df.loc[output_df['ligand_A_407_VF1_interactions'] == 'contact', 'ligand_numeric'] = 1
output_df.loc[output_df['ligand_A_407_VF1_interactions'] == 'binding site', 'ligand_numeric'] = 2
output_df.loc[output_df['ligand_A_407_VF1_interactions'] == 'second shell', 'ligand_numeric'] = 3
create_chimerax_file(output_df=output_df,
                    output_dir=f'{plot_dir}',
                    attribute_name='ligand_numeric',
                    descriptive_text='Ligand A 407 VF1 Interactions')

bonds_df = pd.read_csv(f'{plot_dir}/5C1M_MOR_bonds.csv')
#bonds_df = bonds_df.loc[np.logical_and(bonds_df.chain == 'A', bonds_df.protein_protein), :]
bonds_df = bonds_df.loc[bonds_df.chain == 'A', :]
subset = bonds_df.loc[bonds_df.resi_struct.isin(range(146, 148)), :]
subset = subset.loc[subset.bond_type == 'hbond', :]


# Generate chimerax file for kyte_doolittle
output_df = scores.drop_duplicates(subset=['chain', 'resi_struct', 'resn_struct'])
output_df = output_df[['chain', 'resi_struct', 'resn_struct', 'kyte_doolittle']]
output_df = output_df.loc[output_df.resi_struct.notna(), :]
create_chimerax_file(output_df=output_df,
                    output_dir=f'{plot_dir}',
                    attribute_name='kyte_doolittle',
                    descriptive_text='Kyte Doolittle')
