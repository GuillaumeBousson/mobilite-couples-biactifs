# ---------------------------------------------------------------------------
# 03 - Traduire les résultats et tracer les figures
#
# Un coefficient de logit ne se lit pas directement. Pour les décisions
# d'équipement (niveaux 2 et 3), je calcule donc ce que le modèle prédit si tous
# les couples avaient une place de stationnement, puis si aucun n'en avait, en
# laissant tout le reste inchangé.
# ---------------------------------------------------------------------------
import json
import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, "code")
from modele import level2
from outils_estimation import preparer

df = pd.read_parquet("donnees/couples_biactifs_estimes.parquet")
par = json.load(open("resultats/parametres.json"))
coef = pd.read_csv("resultats/coefficients.csv", sep=";", decimal=",")
lire = lambda nom: pd.read_csv(f"resultats/{nom}.csv", sep=";", decimal=",", index_col=0)

# --- 1. Effet prédit d'une variable du logement sur le nombre de voitures -------
def predire(df, **scenario):
    """Parts prédites de couples sans voiture, avec une voiture, avec deux ou plus."""
    d = df.assign(**scenario)
    p2, p3 = par["niveau2"], par["niveau3"]
    x = lambda p, s: p[f"B_ENF_{s}"] * d.ENF_0_16 + p[f"B_PROPRIO_{s}"] * d.HOMEOWNERSHIP \
        + p[f"B_PARKING_{s}"] * d.PARKING + p[f"B_GC_{s}"] * d.RES_OUTERRING
    # niveau 2 : deuxième voiture, pour un couple qui en a au moins une
    ecart2 = (p2["ASC_2c_l2"] + x(p2, "l2") + d.LS_2c - d.LS_1c) / p2["sigma_l2"]
    p_deux = 1 / (1 + np.exp(-ecart2))
    ls_car = level2(list(p2.values()), preparer(d), logsum=True)
    # niveau 3 : au moins une voiture
    ecart3 = (p3["ASC_car_l3"] + x(p3, "l3") + ls_car - d.V_BB) / p3["sigma_l3"]
    p_voiture = 1 / (1 + np.exp(-ecart3))
    w = d.WEIGHT_hh
    moyenne = lambda s: round(100 * float((s * w).sum() / w.sum()), 1)
    return {"aucune_%": moyenne(1 - p_voiture), "une_%": moyenne(p_voiture * (1 - p_deux)),
            "deux_ou_plus_%": moyenne(p_voiture * p_deux)}

scenarios = {
    "Prédiction du modèle (situation observée)": {},
    "Si tous les couples avaient une place": {"PARKING": 1},
    "Si aucun couple n'avait de place": {"PARKING": 0},
    "Si tous étaient propriétaires": {"HOMEOWNERSHIP": 1},
    "Si aucun n'était propriétaire": {"HOMEOWNERSHIP": 0},
    "Si tous avaient un enfant de 16 ans ou moins": {"ENF_0_16": 1},
    "Si aucun n'en avait": {"ENF_0_16": 0},
}
predictions = pd.DataFrame({nom: predire(df, **s) for nom, s in scenarios.items()}).T
predictions.to_csv("resultats/predictions_equipement.csv", sep=";", decimal=",", encoding="utf-8")
print(predictions)

# Poids de Pareto : poids des préférences de la femme dans les choix du couple
pareto = df.groupby(np.select([df.WOMANmoreEDU == 1, df.MANmoreEDU == 1], ["Elle est plus diplômée", "Il est plus diplômé"],
                              "Même niveau de diplôme")).POIDS_PARETO.agg(["mean", "count"]).round(3)
pareto.to_csv("resultats/poids_pareto.csv", sep=";", decimal=",", encoding="utf-8")
print(pareto)

# --- 2. Figures ------------------------------------------------------------------
BLEU, ORANGE = "#2a78d6", "#eb6834"
BLEUS = ["#86b6ef", "#2a78d6", "#0d366b"]
ENCRE, GRIS, GRILLE, FOND = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": GRILLE, "axes.labelcolor": GRIS,
                     "xtick.color": GRIS, "ytick.color": GRIS, "figure.facecolor": FOND, "axes.facecolor": FOND,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False})
fr = lambda v, d=1: f"{v:.{d}f}".replace(".", ",")


def habiller(fig, ax, titre, sous_titre, source):
    fig.text(0.02, 0.965, titre, fontsize=12.5, fontweight="bold", color=ENCRE, va="top")
    fig.text(0.02, 0.905, sous_titre, fontsize=9.5, color=GRIS, va="top")
    fig.text(0.02, 0.02, source, fontsize=8, color=GRIS, va="bottom")
    ax.tick_params(length=0)


def enregistrer(fig, nom):
    fig.savefig(f"figures/{nom}.png", dpi=200)
    plt.close(fig)


# Figure 1 : ce que retire le filtre
f = lire("filtre_par_departement").drop("Île-de-France").sort_values("part_retiree_%")
fig, ax = plt.subplots(figsize=(8, 4.6))
fig.subplots_adjust(left=0.2, right=0.93, top=0.82, bottom=0.13)
ax.barh(f.index, f["part_retiree_%"], color=BLEU, height=0.6)
for y, v in enumerate(f["part_retiree_%"]):
    ax.text(v + 1.2, y, fr(v, 0) + " %", va="center", color=ENCRE, fontsize=9.5)
ax.set_xlim(0, 110); ax.set_xticks([]); ax.spines["bottom"].set_visible(False)
habiller(fig, ax, "Le filtre retire tous les couples parisiens, et trois sur dix ailleurs",
         "Part des couples bi-actifs écartés parce qu'un conjoint au moins travaille dans sa commune de résidence",
         "Source : Insee, recensement de la population 2015. Champ : couples bi-actifs d'Île-de-France. Pourcentages pondérés.")
enregistrer(fig, "fig1_filtre_par_departement")

# Figure 2 : modes de transport, hommes et femmes
m = lire("modes_par_sexe").loc[["Voiture", "Transports en commun", "Deux-roues", "Marche"]]
fig, ax = plt.subplots(figsize=(8, 4.6))
fig.subplots_adjust(left=0.06, right=0.97, top=0.80, bottom=0.17)
x = np.arange(len(m)); l = 0.36
for decalage, col, couleur, nom in ((-l / 2 - 0.01, "hommes", BLEU, "Hommes"), (l / 2 + 0.01, "femmes", ORANGE, "Femmes")):
    ax.bar(x + decalage, m[col], width=l, color=couleur, label=nom)
    for xi, v in zip(x + decalage, m[col]):
        ax.text(xi, v + 1, fr(v, 0) + " %", ha="center", color=ENCRE, fontsize=9.5)
ax.set_xticks(x); ax.set_xticklabels(m.index, fontsize=10, color=ENCRE)
ax.set_yticks([]); ax.set_ylim(0, 62)
ax.legend(frameon=False, loc="upper right", fontsize=9.5, labelcolor=ENCRE)
habiller(fig, ax, "Lui prend plus souvent la voiture, elle les transports en commun",
         "Mode de transport pour aller travailler, selon le sexe",
         "Source : Insee, recensement 2015. Champ : couples bi-actifs d'Île-de-France hors Paris, dont aucun conjoint ne travaille\ndans sa commune de résidence (173 560 couples). Pourcentages pondérés.")
enregistrer(fig, "fig2_modes_par_sexe")

# Figure 3 : nombre de voitures selon le logement
v = lire("voitures_selon_logement")
lignes = ["Petite couronne - sans place", "Petite couronne - avec place", "Grande couronne - sans place", "Grande couronne - avec place"]
etiquettes = ["Petite couronne,\nsans place", "Petite couronne,\navec place", "Grande couronne,\nsans place", "Grande couronne,\navec place"]
v = v.loc[lignes[::-1]]
fig, ax = plt.subplots(figsize=(8, 4.6))
fig.subplots_adjust(left=0.2, right=0.97, top=0.76, bottom=0.13)
gauche = np.zeros(len(v))
for col, couleur, nom in (("aucune_%", BLEUS[0], "Aucune voiture"), ("une_%", BLEUS[1], "Une voiture"), ("deux_ou_plus_%", BLEUS[2], "Deux ou plus")):
    ax.barh(range(len(v)), v[col], left=gauche, color=couleur, height=0.6, edgecolor=FOND, linewidth=2, label=nom)
    for y, (g, val) in enumerate(zip(gauche, v[col])):
        if val >= 6:
            ax.text(g + val / 2, y, fr(val, 0) + " %", ha="center", va="center", fontsize=9.5,
                    color=ENCRE if couleur == BLEUS[0] else "white")
    gauche += v[col].to_numpy()
ax.set_yticks(range(len(v))); ax.set_yticklabels(etiquettes[::-1], color=ENCRE)
ax.set_xticks([]); ax.set_xlim(0, 100); ax.spines["bottom"].set_visible(False)
ax.legend(frameon=False, ncol=3, loc="lower left", bbox_to_anchor=(-0.01, 1.0), fontsize=9.5, labelcolor=ENCRE, handlelength=1.2)
habiller(fig, ax, "Sans place de stationnement, on se passe bien plus souvent de voiture",
         "Nombre de voitures du couple, selon le lieu de résidence et la présence d'une place de stationnement au logement",
         "Source : Insee, recensement 2015. Champ : couples bi-actifs d'Île-de-France hors Paris (173 560 couples). Pourcentages pondérés.")
enregistrer(fig, "fig3_voitures_selon_logement")

# Figure 4 : part de couples sans voiture prédite par le modèle, selon trois caractéristiques
lignes = [("Place de stationnement\nau logement", "Si aucun couple n'avait de place", "Si tous les couples avaient une place"),
          ("Propriétaire\ndu logement", "Si aucun n'était propriétaire", "Si tous étaient propriétaires"),
          ("Enfant de 16 ans\nou moins", "Si aucun n'en avait", "Si tous avaient un enfant de 16 ans ou moins")]
fig, ax = plt.subplots(figsize=(8, 4.6))
fig.subplots_adjust(left=0.22, right=0.95, top=0.76, bottom=0.2)
for y, (nom, sans, avec) in enumerate(lignes[::-1]):
    a, b = predictions.loc[sans, "aucune_%"], predictions.loc[avec, "aucune_%"]
    ax.plot([a, b], [y, y], color=GRILLE, linewidth=3, zorder=1)
    ax.scatter([a], [y], s=90, color=ORANGE, zorder=2, label="Si aucun couple ne l'avait" if y == 0 else None)
    ax.scatter([b], [y], s=90, color=BLEU, zorder=2, label="Si tous les couples l'avaient" if y == 0 else None)
    ax.text(a + 0.35, y, fr(a) + " %", va="center", fontsize=9.5, color=ENCRE)
    ax.text(b - 0.35, y, fr(b) + " %", va="center", ha="right", fontsize=9.5, color=ENCRE)
ax.set_yticks(range(len(lignes))); ax.set_yticklabels([l[0] for l in lignes[::-1]], color=ENCRE)
ax.set_ylim(-0.6, len(lignes) - 0.4); ax.set_xlim(0, 16)
ax.set_xticks(range(0, 17, 4)); ax.set_xticklabels([f"{t} %" for t in range(0, 17, 4)])
ax.xaxis.grid(True, color=GRILLE, linewidth=0.6); ax.set_axisbelow(True)
ax.set_xlabel("Part de couples sans voiture prédite par le modèle", fontsize=9)
ax.legend(frameon=False, ncol=2, loc="lower left", bbox_to_anchor=(-0.01, 1.0), fontsize=9.5, labelcolor=ENCRE, handletextpad=0.2)
habiller(fig, ax, "À profil égal, la place de stationnement est ce qui sépare le plus les couples",
         "Part de couples sans voiture selon le modèle, quand on change une seule caractéristique pour tous les couples",
         "Modèle logit emboîté estimé sur 173 560 couples bi-actifs d'Île-de-France hors Paris (recensement 2015).\nPart observée : 6,7 %. Lecture : ces écarts sont des associations, pas des effets causaux.")
enregistrer(fig, "fig4_couples_sans_voiture_predits")
print("Figures enregistrées dans figures/")
