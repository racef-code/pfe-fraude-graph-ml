"""Étape 3 (STUB) : construction du graphe fiscal hétérogène depuis 3 tables.
Corps à implémenter quand les vraies données arrivent (Étape 4)."""
import pandas as pd
from torch_geometric.data import HeteroData

# Relations attendues (cahier des charges §4.1)
RELATION_TYPES = [
    "client_fournisseur",
    "partage_dirigeant",
    "partage_adresse",
    "partage_comptable",
    "participation",
]
# Types de nœuds
NODE_TYPES = ["entreprise", "particulier"]


def build_fiscal_graph(nodes_df: pd.DataFrame, edges_df: pd.DataFrame) -> HeteroData:
    """Construit un HeteroData depuis les tables Contribuables + Relations.

    nodes_df: id, type (entreprise/particulier), features (CA/revenu, secteur,
        ancienneté, effectif, impôt déclaré, ratios, retards, rectifications, écarts),
        label (fraudeur/non-fraudeur, enquêtés seulement -> NaN sinon).
    edges_df: id_source, id_cible, type_relation (cf. RELATION_TYPES).

    TODO (Étape 4, données réelles) :
      1. Séparer les nœuds par type ; encoder/normaliser les features par type.
      2. Indexer les ids -> indices contigus par type de nœud.
      3. Pour chaque type de relation -> HeteroData[src, rel, dst].edge_index.
      4. Masque de labels = uniquement les contribuables enquêtés
         (biais de sélection : non-enquêtés != négatifs).
      5. Splits train/val/test stratifiés sur les seuls nœuds labellisés.
    """
    raise NotImplementedError(
        "Stub fiscal : à compléter en Étape 4 avec les vraies données."
    )
