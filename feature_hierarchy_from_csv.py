from dataclasses import dataclass

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


@dataclass(frozen=True)
class ClassificationRule:
    group: str
    subcategory: str
    exact_names: frozenset[str] = frozenset()
    prefixes: tuple[str, ...] = ()

    def matches(self, feature_name: str) -> bool:
        return feature_name in self.exact_names or feature_name.startswith(self.prefixes)


PALETTE = {
    "Sequence-derived": "#4C78A8",
    "Single-residue": "#72B7B2",
    "Interactions": "#F58518",
    "Secondary/Tertiary": "#54A24B",
    "Protein-ligand": "#B279A2",
    "Unclassified": "#9D9D9D",
}

METADATA_COLUMNS = {
    "name",
    "chain",
    "resi_mut",
    "resn_mut",
    "resm",
    "type",
    "resi_struct",
    "resn_struct",
    "struct_info",
    "mut_info",
}

RULES = [
    ClassificationRule(
        group="Sequence-derived",
        subcategory="Mutation effects",
        exact_names=frozenset(
            {
                "effect",
                "pos_effect",
                "effect_quartile",
                "effect_variance",
                "effect_variance_rank",
                "effect_ranking",
                "avg_effect",
                "avg_effect_quartile",
                "mutation_category",
                "total_lof",
                "total_gof",
            }
        ),
    ),
    ClassificationRule(
        group="Sequence-derived",
        subcategory="Substitution scores",
        exact_names=frozenset({"blosum90", "phat_score"}),
    ),
    ClassificationRule(
        group="Sequence-derived",
        subcategory="Amino-acid group labels",
        exact_names=frozenset(
            {"wildtype_aa_group", "mut_aa_group", "wildtype_mut_aa_group"}
        ),
    ),
    ClassificationRule(
        group="Sequence-derived",
        subcategory="Kidera factors",
        prefixes=("kidera_",),
    ),
    ClassificationRule(
        group="Sequence-derived",
        subcategory="AAIndex descriptors",
        prefixes=("KYTJ820101", "WIMW960101", "ENGD860101", "FAUJ880103", "KLEP840101", "GRAR740102", "CHAM820101", "BHAR880101"),
    ),
    ClassificationRule(
        group="Sequence-derived",
        subcategory="Sliding window averages",
        prefixes=("sequence_window_",),
    ),
    ClassificationRule(
        group="Single-residue",
        subcategory="SASA",
        exact_names=frozenset(
            {
                "sasa",
                "sasa_backbone",
                "sasa_sidechain",
                "sasa_polar",
                "sasa_nonpolar",
            }
        ),
    ),
    ClassificationRule(
        group="Single-residue",
        subcategory="Hydrophobicity",
        exact_names=frozenset({"kyte_doolittle"}),
    ),
    ClassificationRule(
        group="Interactions",
        subcategory="Hydrogen bonds",
        exact_names=frozenset(
            {"bb_hbond_count", "sc_hbond_count", "total_hbond_count"}
        ),
    ),
    ClassificationRule(
        group="Interactions",
        subcategory="Electrostatic and covalent contacts",
        exact_names=frozenset(
            {"salt_bridge_count", "ionic_bond_count", "disulfide_bond_count"}
        ),
    ),
    ClassificationRule(
        group="Interactions",
        subcategory="Aromatic and van der Waals contacts",
        exact_names=frozenset(
            {"pi_stacking_count", "cation_pi_count", "vdw_contact_count"}
        ),
    ),
    ClassificationRule(
        group="Interactions",
        subcategory="Aggregate bond summaries",
        exact_names=frozenset(
            {"total_bond_count", "total_within_chain_bonds", "total_between_chain_bonds"}
        ),
    ),
    ClassificationRule(
        group="Secondary/Tertiary",
        subcategory="Within structure distances",
        exact_names=frozenset(
            {
                "distance_to_nearest_surface_residue",
                "distance_from_membrane_edge",
                "distance_to_center_of_mass",
            }
        ),
    ),

    ClassificationRule(
        group="Interactions",
        subcategory="DSSP bond energies",
        prefixes=("dssp_nh", "dssp_o"),
    ),
    ClassificationRule(
        group="Secondary/Tertiary",
        subcategory="DSSP bond angles",
        exact_names=frozenset({"dssp_acc", "dssp_tco", "dssp_phi", "dssp_psi", "dssp_kappa", "dssp_alpha"}),
    ),
    ClassificationRule(
        group="Secondary/Tertiary",
        subcategory="Secondary structure descriptors",
        exact_names=frozenset({"ss_domain_length"}),
        prefixes=("ss_domain_log2",),
    ),
    ClassificationRule(
        group="Secondary/Tertiary",
        subcategory="Secondary structure feature averages",
        prefixes=("ss_domain_",),
    ),
    ClassificationRule(
        group="Secondary/Tertiary",
        subcategory="Secondary structure labels",
        exact_names=frozenset({"ss_group", "ss_domains"}),
    ),
    ClassificationRule(
        group="Secondary/Tertiary",
        subcategory="Packing",
        prefixes=("packing_",),
    ),
    ClassificationRule(
        group="Secondary/Tertiary",
        subcategory="Neighboring residue counts",
        exact_names=frozenset({"n_same_chain_neighbors", "n_different_chain_neighbors", "n_neighbors"}),
    ),
    ClassificationRule(
        group="Secondary/Tertiary",
        subcategory="Neighboring residue diversity",
        exact_names=frozenset({"neighbor_aa_entropy", "neighbor_aa_group_entropy"}),
    ),
    ClassificationRule(
        group="Secondary/Tertiary",
        subcategory="Neighborhood secondary structure composition",
        exact_names=frozenset({"secondary_structure_coarse_entropy", "secondary_structure_granular_entropy", "neighbor_prop_alpha_helix", "neighbor_prop_beta_sheet", "neighbor_prop_coil"}),
    ),
    ClassificationRule(
        group="Secondary/Tertiary",
        subcategory="Neighborhood feature averages",
        prefixes=("neighborhood_",),
    ),
    ClassificationRule(
        group="Secondary/Tertiary",
        subcategory="Sequence distance of neighboring residues",
        exact_names=frozenset({"prop_long_range_neighbors", "mean_neighbor_sequence_distance"}),
    ),
    ClassificationRule(
        group="Secondary/Tertiary",
        subcategory="Graph-based connectivity",
        prefixes=("graph_",),
    ),
    ClassificationRule(
        group="Protein-ligand",
        subcategory="Ligand contact definitions",
        prefixes=("ligand_",),
    ),
]
def extract_feature_columns(scores_df: pd.DataFrame) -> list[str]:
    return [
        column
        for column in scores_df.columns
        if column not in METADATA_COLUMNS and not column.startswith("Unnamed:")
    ]


def classify_feature(
    feature_name: str,
    rules: list[ClassificationRule] = RULES,
) -> tuple[str, str]:
    for rule in rules:
        if rule.matches(feature_name):
            return rule.group, rule.subcategory

    return "Unclassified", "Unclassified"


def build_assignment_dataframe(
    feature_columns: list[str],
    rules: list[ClassificationRule] = RULES,
) -> pd.DataFrame:
    assignments_df = pd.DataFrame({"feature_name": feature_columns})
    assignments_df[["group", "subcategory"]] = assignments_df["feature_name"].apply(
        lambda feature_name: pd.Series(classify_feature(feature_name, rules=rules))
    )
    return assignments_df


def aggregate_counts(assignments_df: pd.DataFrame) -> pd.DataFrame:
    counts_df = (
        assignments_df.groupby(["group", "subcategory"], as_index=False)
        .size()
        .rename(columns={"size": "feature_count"})
    )
    counts_df["group_count"] = counts_df.groupby("group")["feature_count"].transform("sum")
    counts_df = counts_df.sort_values(
        ["group_count", "feature_count", "group", "subcategory"],
        ascending=[False, False, True, True],
    ).reset_index(drop=True)
    counts_df["subcategory"] = pd.Categorical(
        counts_df["subcategory"],
        categories=counts_df["subcategory"].tolist(),
        ordered=True,
    )
    return counts_df


def plot_grouped_horizontal_bars(
    counts_df: pd.DataFrame,
    title: str,
    ax: plt.Axes | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    if counts_df.empty:
        raise ValueError("No classified feature rows available to plot.")

    if ax is None:
        fig_height = max(6.0, len(counts_df) * 0.5 + 1.5)
        fig, ax = plt.subplots(figsize=(13, fig_height))
    else:
        fig = ax.figure

    sns.barplot(
        data=counts_df,
        x="feature_count",
        y="subcategory",
        hue="group",
        order=counts_df["subcategory"].tolist(),
        hue_order=counts_df["group"].drop_duplicates().tolist(),
        palette=PALETTE,
        dodge=False,
        ax=ax,
    )
    ax.set_xlabel("Number of features")
    ax.set_ylabel("")
    ax.set_title(title, fontsize=16, pad=12)
    fig.tight_layout()
    return fig, ax


def build_feature_hierarchy_data(
    scores_df: pd.DataFrame,
    rules: list[ClassificationRule] = RULES,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    feature_columns = extract_feature_columns(scores_df)
    assignments_df = build_assignment_dataframe(feature_columns, rules=rules)
    counts_df = aggregate_counts(assignments_df)
    return assignments_df, counts_df


save_dir = '/Users/ngreenwald/Library/CloudStorage/Box-Box/WCM Lab/Noah/feature_pipeline/fig2'
scores = pd.read_csv(f'{save_dir}/5C1M_features.csv')
assignments_df, counts_df = build_feature_hierarchy_data(scores, rules=RULES)
fig, ax = plot_grouped_horizontal_bars(counts_df, title="Feature hierarchy")


fig.savefig(f'{save_dir}/feature_hierarchy.png')
plt.close(fig)
