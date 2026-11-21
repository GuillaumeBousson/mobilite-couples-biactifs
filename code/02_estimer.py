# ---------------------------------------------------------------------------
# 02 - Estimer le modèle, niveau par niveau
#
# Quatre décisions emboîtées, estimées de la plus fine à la plus générale :
#   niveau 0 : sans voiture, quel mode (transports en commun, marche, deux-roues) ?
#   niveau 1 : qui, dans le couple, prend la voiture ?
#   niveau 2 : une ou deux voitures ?
#   niveau 3 : une voiture ou aucune ?
# Chaque niveau transmet au suivant un résumé de ce que valent ses options (le "logsum").
#
# À chaque niveau, j'estime d'abord un modèle de base (constantes, temps et distances),
# puis le modèle complet, qui ajoute les caractéristiques du couple et de son logement.
# Un test du rapport de vraisemblance dit si ces variables améliorent l'ajustement.
# ---------------------------------------------------------------------------
import json
import sys
import warnings

import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, "code")
from modele import level0, level1, level2, level3
from outils_estimation import estimer, noms_parametres, preparer, test_rapport_vraisemblance

df = pd.read_parquet("donnees/couples_biactifs_prepares.parquet")

# Dans le recensement, on sait que les deux conjoints vont travailler en voiture, pas qui conduit
# l'autre. Les paramètres "passager" ne peuvent donc pas être séparés des paramètres "conducteur" :
# je les fixe à zéro. Sans cela, quatre paramètres du niveau 1 ne sont pas identifiés.
NORMALISATION = {"ASC_Cp_m_l1": 0.0, "ASC_Cp_w_l1": 0.0, "delta_Cp_m_l1": 0.0, "delta_Cp_w_l1": 0.0}

VARIABLES_DU_COUPLE = {
    0: [p for p in noms_parametres(level0) if p.startswith(("B_AGE", "B_CADRE"))],
    1: ["L_FEMME_PLUS_DIPL_l1", "L_HOMME_PLUS_DIPL_l1", "B_ENF_B_m_l1", "B_ENF_B_w_l1"],
    2: ["B_ENF_l2", "B_PROPRIO_l2", "B_PARKING_l2", "B_GC_l2"],
    3: ["B_ENF_l3", "B_PROPRIO_l3", "B_PARKING_l3", "B_GC_l3"],
}
FONCTIONS = {0: level0, 1: level1, 2: level2, 3: level3}
tableaux, bilans, parametres = [], [], {}

for niveau in range(4):
    func = FONCTIONS[niveau]
    data = preparer(df)
    fixes = NORMALISATION if niveau == 1 else {}

    print(f"\nNiveau {niveau}, modèle de base")
    p_base, t_base, b_base = estimer(func, data, fixes={**fixes, **{a: 0.0 for a in VARIABLES_DU_COUPLE[niveau]}})
    print(f"Niveau {niveau}, modèle complet")
    p, t, b = estimer(func, data, fixes=fixes, depart=p_base)
    stat, ddl, pval = test_rapport_vraisemblance(b_base, b)
    print(f"  test du rapport de vraisemblance : {stat:.0f} pour {ddl} paramètres, p = {pval:.2g}")

    for nom, tab, bil in (("base", t_base, b_base), ("complet", t, b)):
        tableaux.append(tab.assign(niveau=niveau, modele=nom))
        bilans.append({"niveau": niveau, "modele": nom, **bil})
    bilans[-1].update({"test_rv": stat, "test_rv_ddl": ddl, "test_rv_p": pval})
    parametres[f"niveau{niveau}"] = p

    # Ce que le niveau transmet au suivant
    valeurs = list(p.values())
    if niveau == 0:
        ls_m, ls_w = level0(valeurs, data, logsum=True)
        df["LS_m"], df["LS_w"] = ls_m.detach().numpy(), ls_w.detach().numpy()
    elif niveau == 1:
        df["LS_1c"], df["LS_2c"] = level1(valeurs, data, logsum=True)
        df["V_BB"] = level1(valeurs, data, utility_nestB=True)
        df["POIDS_PARETO"] = level1(valeurs, data, pareto_weight=True).detach().numpy()
    elif niveau == 2:
        df["LS_car"] = level2(valeurs, data, logsum=True)

pd.concat(tableaux)[["niveau", "modele", "parametre", "coefficient", "ecart_type", "z", "p_valeur", "fixe"]] \
    .to_csv("resultats/coefficients.csv", sep=";", decimal=",", index=False, encoding="utf-8")
pd.DataFrame(bilans).to_csv("resultats/qualite_ajustement.csv", sep=";", decimal=",", index=False, encoding="utf-8")
json.dump(parametres, open("resultats/parametres.json", "w"), indent=1)
df.to_parquet("donnees/couples_biactifs_estimes.parquet")
print("\nTerminé : resultats/coefficients.csv et resultats/qualite_ajustement.csv")
