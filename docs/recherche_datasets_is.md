# Recherche — datasets et direction IS / graphe hétérogène

Date : 2026-06-17

## Conclusion courte

Pour le PFE, la direction **graphe hétérogène orienté entreprises / IS** est la bonne. YelpChi reste utile comme banc d'essai Graph ML public, mais il ne doit pas être présenté comme dataset fiscal. Le vrai modèle final doit cibler le nœud `company` et exploiter les relations fiscales : transactions, dirigeant commun, adresse commune, comptable commun, secteur, participation/contrôle.

## Références utiles trouvées

| Source | Utilité pour le PFE | Remarque |
|---|---|---|
| **TED: Related Party Transaction guided Tax Evasion Detection on Heterogeneous Graph** — arXiv/GitHub `yimingxu24/TED` | Très proche du sujet : tax evasion, entreprises, graphe hétérogène, related-party transactions, attention hiérarchique. | Le repo annonce deux datasets fiscaux hétérogènes `T20H` et `T15S`. À examiner comme candidat si les données sont réellement téléchargeables/utilisables. |
| **Eagle: edge feature aware heterogeneous GNN for tax evasion detection** | Confirme que la formulation tax-evasion = graphe hétérogène avec features d'arêtes est pertinente. | Article ScienceDirect difficile à extraire automatiquement, mais le résumé indique un vrai dataset fiscal non public. |
| **Corporate Fraud Detection in Rich-yet-Noisy Financial Graph / KeHGN-R** — arXiv/GitHub `wangskyGit/KeHGN-R` | Dataset d'entreprises avec relations DSE/RPT, labels de fraude corporate, problème proche de sociétés/contrôle/transactions. | Données CSMAR via Google Drive ; pas exactement fiscal IS, mais beaucoup plus proche que YelpChi. Vérifier licence/conditions. |
| **GLC-GNN** | Utile pour discuter hétérophilie/camouflage et limites des GNN fraud classiques. | Plus général fraud/anomaly, pas fiscal. |
| **Amazon Fraud Dataset Benchmark (FDB)** | Bon benchmark général de fraude tabulaire. | Pas fiscal ni relationnel entreprise par défaut. |

## Choix recommandé maintenant

1. **Court terme (avant données réelles)**
   - Garder YelpChi pour valider GraphSAGE/GAT/multi-relation et métriques.
   - Ajouter un banc synthétique IS hétérogène pour tester le pipeline fiscal `HeteroData` sans inventer de faux résultat scientifique.
   - Explorer TED `T20H/T15S` et KeHGN-R comme datasets publics/semi-publics plus proches des entreprises.

2. **Quand les données réelles arrivent**
   - Tables minimales attendues :
     - `nodes_df`: `id`, `type`, features numériques, `label` pour `company` uniquement si enquêté.
     - `edges_df`: `id_source`, `id_cible`, `type_relation`.
   - Types de nœuds cibles/support : `company`, `person`, `address`, `accountant`, `sector`.
   - Relations prioritaires : `transaction`, `has_director`, `registered_at`, `uses_accountant`, `in_sector`, `participates_in`.

## Point méthodologique important

Les entreprises non enquêtées ne doivent pas être codées comme `label=0`. Elles doivent rester `label=NaN` / inconnues et être exclues des masks train/val/test. Sinon le modèle apprendrait un faux signal et les métriques seraient optimistes.
