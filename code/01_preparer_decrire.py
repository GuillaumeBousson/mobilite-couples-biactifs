# ---------------------------------------------------------------------------
# 01 - Préparer les données et les décrire
#
# Source : recensement de la population 2015, Île-de-France, couples dont les
# deux conjoints travaillent. Le fichier compte une ligne par couple, avec les
# temps de trajet en voiture et en transports en commun de chaque conjoint.
# Il n'est pas dans le dépôt (voir donnees/LISEZ-MOI.md).
#
# Ce script fait trois choses :
#   1. il regarde ce que retire le filtre des "intrazonaux" ;
#   2. il construit les variables utilisées dans le modèle ;
#   3. il produit les tableaux descriptifs de la note.
# Tous les pourcentages sont pondérés par les poids du recensement.
# ---------------------------------------------------------------------------
import pandas as pd

brut = pd.read_parquet("donnees/couple_biactives_2015.parquet")
brut["poids"] = (brut.IPONDI_m + brut.IPONDI_w) / 2          # poids du couple

MODES = {"CAR": "Voiture", "PUBLIC": "Transports en commun", "TWO_WHEELS": "Deux-roues", "WALK": "Marche"}
DEPARTEMENTS = {"75": "Paris", "92": "Hauts-de-Seine", "93": "Seine-Saint-Denis", "94": "Val-de-Marne",
                "77": "Seine-et-Marne", "78": "Yvelines", "91": "Essonne", "95": "Val-d'Oise"}


def part(df, colonne, poids):
    """Répartition pondérée d'une variable, en %."""
    return (100 * df.groupby(colonne)[poids].sum() / df[poids].sum()).round(1)


def ecrire(tableau, nom):
    tableau.to_csv(f"resultats/{nom}.csv", sep=";", decimal=",", encoding="utf-8")
    print(f"\n{nom}\n{tableau}")


# --- 1. Ce que retire le filtre -----------------------------------------------
# Un couple est "intrazonal" dès qu'un des deux conjoints a pour commune de travail
# sa commune de résidence. Les temps de trajet du fichier n'ont alors pas de sens.
brut["meme_commune_h"] = brut.DCLT_m == brut.COMMUNE
brut["meme_commune_f"] = brut.DCLT_w == brut.COMMUNE
assert ((brut.meme_commune_h | brut.meme_commune_f) == (brut.INTRAZONAL == 1)).all()

filtre = brut.groupby("DEPT").apply(lambda g: pd.Series({
    "couples_fichier": len(g),
    "couples_ponderes": round(g.poids.sum()),
    "part_retiree_%": round(100 * (g.poids * g.INTRAZONAL).sum() / g.poids.sum(), 1)}))
filtre.index = filtre.index.map(DEPARTEMENTS)
filtre.loc["Île-de-France"] = [len(brut), round(brut.poids.sum()),
                               round(100 * (brut.poids * brut.INTRAZONAL).sum() / brut.poids.sum(), 1)]
ecrire(filtre, "filtre_par_departement")

df = brut[brut.INTRAZONAL == 0].reset_index(drop=True)

# Le filtre change-t-il le portrait des déplacements ?
avant_apres = pd.DataFrame({
    "hommes_avant": part(brut, "COMMUTE_MODE_m", "IPONDI_m"), "hommes_apres": part(df, "COMMUTE_MODE_m", "IPONDI_m"),
    "femmes_avant": part(brut, "COMMUTE_MODE_w", "IPONDI_w"), "femmes_apres": part(df, "COMMUTE_MODE_w", "IPONDI_w")})
avant_apres.index = avant_apres.index.map(MODES)
ecrire(avant_apres, "modes_avant_apres_filtre")

trajets = pd.DataFrame({
    "avant": [(brut.TT_VP_m * brut.IPONDI_m).sum() / brut.IPONDI_m.sum() * 60, (brut.DISTANCE_m * brut.IPONDI_m).sum() / brut.IPONDI_m.sum(),
              100 * (brut.poids * (brut.VOIT == 0)).sum() / brut.poids.sum()],
    "apres": [(df.TT_VP_m * df.IPONDI_m).sum() / df.IPONDI_m.sum() * 60, (df.DISTANCE_m * df.IPONDI_m).sum() / df.IPONDI_m.sum(),
              100 * (df.poids * (df.VOIT == 0)).sum() / df.poids.sum()]},
    index=["temps_voiture_hommes_min", "distance_hommes_km", "couples_sans_voiture_%"]).round(1)
ecrire(trajets, "trajets_avant_apres_filtre")

# --- 2. Variables du modèle -----------------------------------------------------
n = len(df)
df["WEIGHT_hh"] = df.poids * n / df.poids.sum()               # poids normalisés : leur somme vaut n
df["WEIGHT_m"] = df.IPONDI_m * n / df.IPONDI_m.sum()
df["WEIGHT_w"] = df.IPONDI_w * n / df.IPONDI_w.sum()
df["AGE40_m"] = (df.AGED_m - 40) / 10                         # âge, en dizaines d'années autour de 40 ans
df["AGE40_w"] = (df.AGED_w - 40) / 10
df["ENF_0_5"] = (df.CHILDREN_0_5 > 0).astype(int)             # au moins un enfant de moins de 6 ans
df["ENF_0_16"] = (df.CHILDREN_0_16 > 0).astype(int)           # au moins un enfant de 16 ans ou moins
df["PARKING"] = (df.GARL == 1).astype(int)                    # le logement a une place de stationnement
df["COURONNE"] = df.RES_OUTERRING.map({0: "Petite couronne", 1: "Grande couronne"})
df.drop(columns=["meme_commune_h", "meme_commune_f"]).to_parquet("donnees/couples_biactifs_prepares.parquet")

# --- 3. Tableaux descriptifs (échantillon filtré) -------------------------------
modes = pd.DataFrame({"hommes": part(df, "COMMUTE_MODE_m", "IPONDI_m"), "femmes": part(df, "COMMUTE_MODE_w", "IPONDI_w")})
modes.index = modes.index.map(MODES)
ecrire(modes, "modes_par_sexe")

# Qui prend la voiture, selon le nombre de voitures et la présence d'un jeune enfant ?
def qui_conduit(g):
    lui, elle = g.COMMUTE_MODE_m == "CAR", g.COMMUTE_MODE_w == "CAR"
    p = lambda masque: round(100 * g.poids[masque].sum() / g.poids.sum(), 1)
    return pd.Series({"les_deux_%": p(lui & elle), "lui_seul_%": p(lui & ~elle), "elle_seule_%": p(~lui & elle),
                      "aucun_%": p(~lui & ~elle), "couples": len(g)})

avec_voiture = df[df.VOIT >= 1]
groupes = [avec_voiture.VOIT.map(lambda v: "Une voiture" if v == 1 else "Deux voitures ou plus"),
           avec_voiture.ENF_0_5.map({0: "sans enfant de moins de 6 ans", 1: "avec enfant de moins de 6 ans"})]
conduite = pd.concat([avec_voiture.groupby(groupes[0]).apply(qui_conduit), avec_voiture.groupby(groupes).apply(qui_conduit)])
conduite.index = [" - ".join(i) if isinstance(i, tuple) else i for i in conduite.index]
ecrire(conduite, "qui_prend_la_voiture")

# Nombre de voitures selon le stationnement et la couronne
def voitures(g):
    return pd.Series({"aucune_%": round(100 * g.poids[g.VOIT == 0].sum() / g.poids.sum(), 1),
                      "une_%": round(100 * g.poids[g.VOIT == 1].sum() / g.poids.sum(), 1),
                      "deux_ou_plus_%": round(100 * g.poids[g.VOIT >= 2].sum() / g.poids.sum(), 1),
                      "couples": len(g)})

motorisation = pd.concat([
    df.groupby(lambda i: "Ensemble").apply(voitures),
    df.groupby("COURONNE").apply(voitures),
    df.groupby(df.PARKING.map({1: "Avec place de stationnement", 0: "Sans place de stationnement"})).apply(voitures),
    df.groupby([df.COURONNE, df.PARKING.map({1: "avec place", 0: "sans place"})]).apply(voitures)
      .rename(index=lambda t: " ".join(t) if isinstance(t, tuple) else t)])
motorisation.index = [" - ".join(i) if isinstance(i, tuple) else i for i in motorisation.index]
ecrire(motorisation, "voitures_selon_logement")

print(f"\nCouples dans le fichier : {len(brut):,} | après filtre : {len(df):,}")
