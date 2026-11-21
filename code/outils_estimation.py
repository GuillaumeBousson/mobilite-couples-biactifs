# ---------------------------------------------------------------------------
# Outils d'estimation : maximum de vraisemblance, écarts-types, tableaux.
#
# Tout est calculé en double précision. En simple précision, la matrice
# hessienne de ce modèle est mal inversée et donne des écarts-types nuls ou
# absents, sans message d'erreur. Le programme vérifie aussi que chaque
# paramètre est identifié, et le signale quand ce n'est pas le cas.
# ---------------------------------------------------------------------------
import inspect
import re

import numpy as np
import pandas as pd
import torch
from scipy import stats
from scipy.optimize import Bounds, minimize


def noms_parametres(func):
    """Lit les noms des paramètres dans le code de la fonction de vraisemblance."""
    source = inspect.getsource(func)
    bloc = re.search(r"\(\s*([^\)]*?)\s*\)\s*=\s*params", source, re.DOTALL).group(1)
    return [n.strip() for n in bloc.replace("\n", "").split(",") if n.strip()]


def preparer(df):
    """Met les variables numériques au format attendu par les fonctions du modèle."""
    v = {c: torch.tensor(df[c].to_numpy(), dtype=torch.float64)
         for c in df.columns if pd.api.types.is_numeric_dtype(df[c])}
    sans_voiture = torch.tensor([2, 3, 5])        # marche, deux-roues, transports en commun
    return {
        "df": df, "vars": v,
        "men_idx": torch.where(torch.isin(v["TRANS_m"], sans_voiture))[0],
        "women_idx": torch.where(torch.isin(v["TRANS_w"], sans_voiture))[0],
        "is_one_car_idx": torch.where(v["VOIT"] == 1)[0],
        "is_multi_car_idx": torch.where(v["VOIT"] >= 2)[0],
        "at_least_one_car_idx": torch.where(v["VOIT"] >= 1)[0],
        "all_idx": torch.arange(len(df)),
    }


def estimer(func, data, fixes=None, depart=None, verbeux=True):
    """Maximise la vraisemblance. `fixes` impose une valeur à certains paramètres."""
    noms = noms_parametres(func)
    fixes = fixes or {}
    libres = [i for i, n in enumerate(noms) if n not in fixes]
    base = torch.tensor([fixes.get(n, 1.0 if n.startswith("sigma") else 0.0) for n in noms], dtype=torch.float64)
    if depart:
        for i, n in enumerate(noms):
            if n in depart and n not in fixes:
                base[i] = depart[n]

    def complet(x):
        p = base.clone()
        p[libres] = x
        return p

    n_obs = len(data["df"])

    def objectif(x):
        # On minimise la log-vraisemblance moyenne : sans cette mise à l'échelle,
        # le premier pas de l'algorithme est démesuré et l'estimation diverge.
        xt = torch.tensor(x, dtype=torch.float64, requires_grad=True)
        perte = func(complet(xt), data, pytorch=True, grad=True) / n_obs
        (g,) = torch.autograd.grad(perte, xt)
        return perte.item(), g.numpy()

    # Deux algorithmes partent du même point et on garde le meilleur : selon le niveau du modèle,
    # l'un ou l'autre peut s'égarer dans une zone où la vraisemblance est plate.
    # Les paramètres d'échelle (sigma) restent strictement positifs.
    bornes = Bounds([0.05 if noms[i].startswith("sigma") else -30.0 for i in libres],
                    [50.0 if noms[i].startswith("sigma") else 30.0 for i in libres])
    x0 = base[libres].numpy()
    essais = []
    for methode, options in (("BFGS", {"gtol": 1e-7, "maxiter": 5000}),
                             ("trust-constr", {"gtol": 1e-7, "xtol": 1e-12, "maxiter": 5000})):
        try:
            r = minimize(objectif, x0, jac=True, method=methode, options=options,
                         bounds=bornes if methode == "trust-constr" else None)
            if np.isfinite(r.fun) and np.isfinite(r.x).all():
                essais.append(r)
        except Exception:
            pass
    res = min(essais, key=lambda r: r.fun)
    # Finition depuis le meilleur point, bornes actives.
    fin = minimize(objectif, np.clip(res.x, bornes.lb, bornes.ub), jac=True, method="trust-constr", bounds=bornes,
                   options={"gtol": 1e-8, "xtol": 1e-13, "maxiter": 3000})
    if np.isfinite(fin.fun) and fin.fun <= res.fun:
        res = fin
    x = torch.tensor(res.x, dtype=torch.float64)
    _, g = objectif(res.x)

    hess = torch.autograd.functional.hessian(lambda z: func(complet(z), data, pytorch=True, grad=True), x)
    hess = ((hess + hess.T) / 2).numpy()
    # Un paramètre non identifié donne une hessienne singulière. On le repère sur la hessienne
    # remise à l'échelle (diagonale à 1), pour ne pas confondre avec un simple problème d'unités.
    diag = np.diag(hess)
    if np.isfinite(hess).all() and (diag > 0).all():
        echelle = 1 / np.sqrt(diag)
        valeurs_propres = np.linalg.eigvalsh(hess * np.outer(echelle, echelle))
    else:
        valeurs_propres = np.array([0.0, 1.0])
    identifie = valeurs_propres.min() > 1e-7
    se = np.sqrt(np.diag(np.linalg.inv(hess))) if identifie else np.full(len(libres), np.nan)

    coef = complet(x).numpy()
    tab = pd.DataFrame({"parametre": noms, "coefficient": coef, "ecart_type": np.nan, "fixe": [n in fixes for n in noms]})
    tab.loc[libres, "ecart_type"] = se
    tab["z"] = tab.coefficient / tab.ecart_type
    tab["p_valeur"] = 2 * (1 - stats.norm.cdf(tab.z.abs()))

    ll = -float(func(coef, data))
    ll0 = -float(func(coef, data, null_loglik=True))
    bilan = {"log_vraisemblance": ll, "log_vraisemblance_nulle": ll0, "parametres_libres": len(libres),
             "rho2_ajuste": 1 - (ll - len(libres)) / ll0, "gradient_max": float(np.abs(g).max()),
             "identifie": bool(identifie), "plus_petite_valeur_propre": float(valeurs_propres.min()),
             "iterations": int(res.nit)}
    if verbeux:
        print(f"  logL = {ll:,.0f} (nulle {ll0:,.0f}) | rho2 = {bilan['rho2_ajuste']:.3f} | gradient max = {bilan['gradient_max']:.2g}"
              f" | identifié : {identifie} (plus petite valeur propre {bilan['plus_petite_valeur_propre']:.2g}) | {res.nit} itérations")
    return dict(zip(noms, coef)), tab, bilan


def test_rapport_vraisemblance(bilan_contraint, bilan_libre):
    """Test du rapport de vraisemblance entre deux modèles emboîtés."""
    stat = 2 * (bilan_libre["log_vraisemblance"] - bilan_contraint["log_vraisemblance"])
    ddl = bilan_libre["parametres_libres"] - bilan_contraint["parametres_libres"]
    return stat, ddl, float(stats.chi2.sf(stat, ddl))
