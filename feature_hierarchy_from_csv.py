from dataclasses import dataclass

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['pdf.fonttype'] = 42


@dataclass(frozen=True)
class ClassificationRule:
    group: str
    subcategory: str
    exact_names: frozenset[str] = frozenset()
    prefixes: tuple[str, ...] = ()

    def matches(self, feature_name: str) -> bool:
        return feature_name in self.exact_names or feature_name.startswith(self.prefixes)


PALETTE = {
    "Mutational effects": "#4C78A8",
    "Residue context": "#72B7B2",
    "Residue bonds": "#F58518",
    "Amino acid sequence-derived": "#54A24B",
    "Spatial averages": "#FFB90F",
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
        group="Mutational effects",
        subcategory="Mutation scores",
        exact_names=frozenset(
            {
                "avg_effect",
                "effect",
                "pos_effect",
                "mutation_category",
            }
        ),
    ),
    ClassificationRule(
        group="Mutational effects",
        subcategory="Mutation score summaries",
        exact_names=frozenset(
            {
                "effect_quartile",
                "effect_variance",
                "effect_variance_rank",
                "effect_ranking",
                "avg_effect_quartile",
                "total_lof",
                "total_gof",
            }
        ),
    ),
    ClassificationRule(
        group="Mutational effects",
        subcategory="Substitution scores",
        exact_names=frozenset({"blosum90", "phat_score"}),
    ),
    ClassificationRule(
        group="Amino acid sequence-derived",
        subcategory="Physico-chemical groups",
        exact_names=frozenset(
            {"wildtype_aa_group", "mut_aa_group", "wildtype_mut_aa_group"}
        ),
    ),
    ClassificationRule(
        group="Amino acid sequence-derived",
        subcategory="Kidera factors",
        prefixes=("kidera_",),
    ),
    ClassificationRule(
        group="Amino acid sequence-derived",
        subcategory="AAIndex descriptors",
        prefixes=("KYTJ820101", "WIMW960101", "ENGD860101", "FAUJ880103", "KLEP840101", "GRAR740102", "CHAM820101", "BHAR880101"),
    ),
    ClassificationRule(
        group="Spatial averages",
        subcategory="Sliding window averages",
        prefixes=("sequence_window_",),
    ),
    ClassificationRule(
        group="Residue context",
        subcategory="SASA",
        exact_names=frozenset(
            {
                "sasa",
                "sasa_backbone",
                "sasa_sidechain",
                "sasa_polar",
                "sasa_nonpolar",
                "kyte_doolittle",
            }
        ),
    ),
    ClassificationRule(
        group="Residue bonds",
        subcategory="Hydrogen bonds",
        exact_names=frozenset(
            {"bb_hbond_count", "sc_hbond_count", "total_hbond_count"}
        ),
    ),
    ClassificationRule(
        group="Residue bonds",
        subcategory="Electrostatic and covalent contacts",
        exact_names=frozenset(
            {"salt_bridge_count", "ionic_bond_count", "disulfide_bond_count"}
        ),
    ),
    ClassificationRule(
        group="Residue bonds",
        subcategory="Aromatic and van der Waals contacts",
        exact_names=frozenset(
            {"pi_stacking_count", "cation_pi_count", "vdw_contact_count"}
        ),
    ),
    ClassificationRule(
        group="Residue bonds",
        subcategory="Aggregate bond summaries",
        exact_names=frozenset(
            {"total_bond_count", "total_within_chain_bonds", "total_between_chain_bonds"}
        ),
    ),
    ClassificationRule(
        group="Residue context",
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
        group="Residue bonds",
        subcategory="DSSP bond energies",
        prefixes=("dssp_nh", "dssp_o"),
    ),
    ClassificationRule(
        group="Residue bonds",
        subcategory="DSSP bond angles",
        exact_names=frozenset({"dssp_acc", "dssp_tco", "dssp_phi", "dssp_psi", "dssp_kappa", "dssp_alpha"}),
    ),
    ClassificationRule(
        group="Residue context",
        subcategory="Secondary structure descriptors",
        exact_names=frozenset({"ss_domain_length"}),
        prefixes=("ss_domain_log2",),
    ),
    ClassificationRule(
        group="Residue context",
        subcategory="Secondary structure labels",
        exact_names=frozenset({"ss_group", "ss_domains"}),
    ),
    ClassificationRule(
        group="Residue context",
        subcategory="Packing",
        prefixes=("packing_",),
    ),
    ClassificationRule(
        group="Residue context",
        subcategory="Neighboring residue counts",
        exact_names=frozenset({"n_same_chain_neighbors", "n_different_chain_neighbors", "n_neighbors"}),
    ),
    ClassificationRule(
        group="Residue context",
        subcategory="Neighb. residue diversity",
        exact_names=frozenset({"neighbor_aa_entropy", "neighbor_aa_group_entropy"}),
    ),
    ClassificationRule(
        group="Residue context",
        subcategory="Neighb. SS composition",
        exact_names=frozenset({"secondary_structure_coarse_entropy", "secondary_structure_granular_entropy", "neighbor_prop_alpha_helix", "neighbor_prop_beta_sheet", "neighbor_prop_coil"}),
    ),
    ClassificationRule(
        group="Spatial averages",
        subcategory="Neighb. avg bonds",
        exact_names=frozenset({
            "neighborhood_bb_hbond_count",
            "neighborhood_cation_pi_count",
            "neighborhood_disulfide_bond_count",
            "neighborhood_dssp_acc",
            "neighborhood_dssp_alpha",
            "neighborhood_dssp_kappa",
            "neighborhood_dssp_nh_o_1_energy",
            "neighborhood_dssp_nh_o_1_relidx",
            "neighborhood_dssp_o_nh_2_energy",
            "neighborhood_dssp_o_nh_2_relidx",
            "neighborhood_dssp_nh_o_2_energy",
            "neighborhood_dssp_nh_o_2_relidx",
            "neighborhood_dssp_o_nh_1_energy",
            "neighborhood_dssp_o_nh_1_relidx",
            "neighborhood_dssp_phi",
            "neighborhood_dssp_psi",
            "neighborhood_dssp_tco",
            "neighborhood_ionic_bond_count",
            "neighborhood_kyte_doolittle",
            "neighborhood_pi_stacking_count",
            "neighborhood_salt_bridge_count",
            "neighborhood_sc_hbond_count",
            "neighborhood_total_between_chain_bonds",
            "neighborhood_total_bond_count",
            "neighborhood_total_hbond_count",
            "neighborhood_total_within_chain_bonds",
            "neighborhood_vdw_contact_count",
            }
        ),
    ),

    ClassificationRule(
        group="Spatial averages",
        subcategory="Neighb. avg mutational effects",
        exact_names=frozenset({
            "neighborhood_avg_effect",
            "neighborhood_blosum90",
            "neighborhood_effect",
            "neighborhood_effect_ranking",
            "neighborhood_effect_variance",
            "neighborhood_effect_variance_rank",
            "neighborhood_phat_score",
        }),
    ),
    ClassificationRule(
        group="Spatial averages",
        subcategory="Neighb. avg context",
        exact_names=frozenset({
            "neighborhood_distance_from_membrane_edge",
            "neighborhood_distance_to_center_of_mass",
            "neighborhood_distance_to_nearest_surface_residue",
            "neighborhood_packing_contact_density",
            "neighborhood_packing_n_atoms",
            "neighborhood_packing_n_neighbor_residues",
            "neighborhood_sasa",
            "neighborhood_sasa_backbone",
            "neighborhood_sasa_nonpolar",
            "neighborhood_sasa_polar",
            "neighborhood_sasa_sidechain",
        }),
    ),

    ClassificationRule(
        group="Spatial averages",
        subcategory="SS avg bonds",
        exact_names=frozenset({
            "ss_domain_bb_hbond_count",
            "ss_domain_cation_pi_count",
            "ss_domain_disulfide_bond_count",
            "ss_domain_dssp_acc",
            "ss_domain_dssp_alpha",
            "ss_domain_dssp_kappa",
            "ss_domain_dssp_nh_o_1_energy",
            "ss_domain_dssp_nh_o_1_relidx",
            "ss_domain_dssp_o_nh_2_energy",
            "ss_domain_dssp_o_nh_2_relidx",
            "ss_domain_dssp_phi",
            "ss_domain_dssp_psi",
            "ss_domain_dssp_tco",
            "ss_domain_ionic_bond_count",
            "ss_domain_kyte_doolittle",
            "ss_domain_pi_stacking_count",
            "ss_domain_salt_bridge_count",
            "ss_domain_sc_hbond_count",
            "ss_domain_total_between_chain_bonds",
            "ss_domain_total_bond_count",
            "ss_domain_total_hbond_count",
            "ss_domain_total_within_chain_bonds",
            "ss_domain_vdw_contact_count",
            "ss_domain_dssp_nh_o_2_energy",
            "ss_domain_dssp_nh_o_2_relidx",
            "ss_domain_dssp_o_nh_1_energy",
            "ss_domain_dssp_o_nh_1_relidx",
            }
        ),
    ),

    ClassificationRule(
        group="Spatial averages",
        subcategory="SS avg mutational effects",
        exact_names=frozenset({
            "ss_domain_avg_effect",
            "ss_domain_blosum90",
            "ss_domain_effect",
            "ss_domain_effect_ranking",
            "ss_domain_effect_variance",
            "ss_domain_effect_variance_rank",
            "ss_domain_phat_score",
        }),
    ),
    ClassificationRule(
        group="Spatial averages",
        subcategory="SS avg context",
        exact_names=frozenset({
            "ss_domain_distance_from_membrane_edge",
            "ss_domain_distance_to_center_of_mass",
            "ss_domain_distance_to_nearest_surface_residue",
            "ss_domain_packing_contact_density",
            "ss_domain_packing_n_atoms",
            "ss_domain_packing_n_neighbor_residues",
            "ss_domain_sasa",
            "ss_domain_sasa_backbone",
            "ss_domain_sasa_nonpolar",
            "ss_domain_sasa_polar",
            "ss_domain_sasa_sidechain",
        }),
    ),

    ClassificationRule(
        group="Residue context",
        subcategory="Sequence dist of neighb. residues",
        exact_names=frozenset({"prop_long_range_neighbors", "mean_neighbor_sequence_distance"}),
    ),

    ClassificationRule(
        group="Residue bonds",
        subcategory="Graph-based connectivity",
        prefixes=("graph_",),
    ),
    ClassificationRule(
        group="Residue bonds",
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



GROUP_ORDER = [
        "Amino acid sequence-derived",
        "Residue bonds",
        "Residue context",
        "Spatial averages",
        "Mutational effects",
        "Unclassified",
]


def plot_grouped_horizontal_bars(
    counts_df: pd.DataFrame,
    title: str,
    ax: plt.Axes | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    if counts_df.empty:
        raise ValueError("No classified feature rows available to plot.")

    if ax is None:
        fig_height = max(6.0, len(counts_df) * 0.5 + 1.5)
        fig, ax = plt.subplots(figsize=(7, fig_height))
    else:
        fig = ax.figure

    sns.barplot(
        data=counts_df,
        x="feature_count",
        y="subcategory",
        hue="group",
        hue_order=GROUP_ORDER,
        order=counts_df["subcategory"].tolist(),
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

    group_rank = {g: i for i, g in enumerate(GROUP_ORDER)}
    counts_df["_group_rank"] = counts_df["group"].map(group_rank)
    counts_df = counts_df.sort_values(["_group_rank", "feature_count"], ascending=[True, False])
    return assignments_df, counts_df


save_dir = '/Users/ngreenwald/Library/CloudStorage/Box-Box/WCM Lab/Noah/feature_pipeline/fig2'
scores = pd.read_csv(f'{save_dir}/5C1M_features.csv')
assignments_df, counts_df = build_feature_hierarchy_data(scores, rules=RULES)
fig, ax = plot_grouped_horizontal_bars(counts_df, title="Feature hierarchy")


fig.savefig(f'{save_dir}/feature_hierarchy2.pdf')
plt.close(fig)
