# Contrat de données — PFE fraude IS par graphe hétérogène

Ce document décrit le format attendu quand les vraies données entreprises / IS arrivent. Le code cible est `src/data/fiscal_graph.py::build_fiscal_graph`.

## 1. Objectif ML

- **Nœud cible :** `company`.
- **Tâche :** classification binaire / score de risque `fraude IS`.
- **Label :** `1 = fraudeur détecté`, `0 = contrôlé/non-fraudeur`, `NaN = inconnu / non enquêté`.
- **Règle critique :** les `NaN` ne sont jamais des négatifs. Ils sont exclus de `train_mask`, `val_mask`, `test_mask`.

## 2. Table `nodes_df`

Colonnes obligatoires :

| colonne | type | description |
|---|---|---|
| `id` | string/int | identifiant anonymisé stable |
| `type` | string | type de nœud : `company`, `person`, `address`, `accountant`, `sector` |

Colonnes optionnelles pour `company` :

| colonne | description |
|---|---|
| `label` | 1/0/NaN |
| `ca` | chiffre d'affaires |
| `resultat` | résultat fiscal/comptable |
| `impot_is` | IS déclaré/payé |
| `marge` | résultat / CA |
| `taux_is_effectif` | IS / résultat imposable approx. |
| `age_entreprise` | ancienneté |
| `retards_declaration` | nombre/score de retards déclaratifs |
| `rectifications` | historique de rectifications |
| `variation_ca` | variation annuelle du CA |
| `ratio_charges` | charges / CA |

Exemple :

```csv
id,type,ca,resultat,impot_is,marge,taux_is_effectif,age_entreprise,retards_declaration,rectifications,variation_ca,ratio_charges,label
c_001,company,1200000,140000,42000,0.1167,0.30,8,0,0,0.08,0.72,0
c_002,company,900000,18000,1000,0.0200,0.055,2,4,2,0.55,0.96,1
p_100,person,,,,,,,,,,,
a_200,address,,,,,,,,,,,
f_300,accountant,,,,,,,,,,,
s_10,sector,,,,,,,,,,,
```

## 3. Table `edges_df`

Colonnes obligatoires :

| colonne | type | description |
|---|---|---|
| `id_source` | string/int | id source, présent dans `nodes_df.id` |
| `id_cible` | string/int | id cible, présent dans `nodes_df.id` |
| `type_relation` | string | type de relation |

Relations recommandées :

| relation | source -> cible | pourquoi c'est utile |
|---|---|---|
| `transaction` | company -> company | facturation / client-fournisseur / flux commerciaux |
| `has_director` | company -> person | dirigeant commun, gérant, bénéficiaire effectif |
| `registered_at` | company -> address | adresse commune / domiciliation |
| `uses_accountant` | company -> accountant | fiduciaire/comptable commun |
| `in_sector` | company -> sector | contrôle de l'effet secteur |
| `participates_in` | company -> company | participation, contrôle, groupe |

Exemple :

```csv
id_source,id_cible,type_relation
c_001,p_100,has_director
c_002,p_100,has_director
c_002,a_200,registered_at
c_001,c_002,transaction
c_002,f_300,uses_accountant
c_001,s_10,in_sector
```

## 4. Prétraitement recommandé

1. **Anonymiser** tous les identifiants avant de les donner au modèle.
2. Garder les features temporelles sous forme robuste : moyennes, ratios, retards, variations, compteurs.
3. Ne pas mélanger périodes : idéalement, construire les features à partir du passé et le label depuis un contrôle ultérieur.
4. Ne pas utiliser de variables qui révèlent directement le contrôle ou la décision administrative finale.
5. Documenter le biais de sélection : les entreprises contrôlées ne sont pas un échantillon aléatoire.

## 5. Commandes de validation

Charger deux CSV réels/anonymisés :

```python
from src.data.real_fiscal import load_fiscal_csv_graph

data = load_fiscal_csv_graph("nodes.csv", "edges.csv", seed=42)
print(data.metadata())
```

Smoke tests :

```bash
python -m pytest tests/test_fiscal_graph.py -q
python -m src.experiments.fiscal_synthetic 3 240 40
```

Le second script est seulement un smoke test synthétique, pas une preuve scientifique.
