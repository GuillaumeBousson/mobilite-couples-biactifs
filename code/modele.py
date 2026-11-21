# ---------------------------------------------------------------------------
# Le modèle : un logit emboîté à quatre niveaux pour les couples bi-actifs
#
#   niveau 3   avoir une voiture ou aucune
#   niveau 2   en avoir une ou deux
#   niveau 1   qui la prend pour aller travailler
#   niveau 0   sans voiture : transports en commun, marche ou deux-roues
#
# Chaque fonction renvoie l'opposé de la log-vraisemblance de son niveau. Elle
# peut aussi renvoyer la vraisemblance d'un modèle sans variable (null_loglik)
# et l'utilité espérée de ses options (logsum), dont le niveau du dessus a besoin.
#
# L'architecture (quatre décisions emboîtées, choix joint du couple pondéré par
# le poids de chaque conjoint) est celle étudiée dans un cours de mon master.
# Les variables qui décrivent le couple et son logement sont mes choix.
#
# Suffixes : _m pour l'homme, _w pour la femme, _l0 à _l3 pour le niveau.
# ---------------------------------------------------------------------------
import torch


def level0(params, df, pytorch=False, null_loglik=False, grad=False, logsum=False):
    """Niveau 0 : mode de transport de chaque conjoint quand il ne prend pas la voiture.

    Trois options : transports en commun (TC), marche (WALK), deux-roues (2R).
    Estimé sur les seuls conjoints qui n'utilisent pas la voiture.
    """
    v = df['vars']
    idx_men = df['men_idx']
    idx_women = df['women_idx']

    def var(name): return v[name]

    if grad:
        params = params.clone().requires_grad_(True)
    else:
        params = torch.tensor(params, dtype=torch.float64)

    (
        sigma_l0,
        ASC_2R_m_l0, ASC_TC_m_l0,

        B_DIST_2R2_m_l0,
        B_DIST_WALK2_m_l0,
        B_TT_TC2_m_l0,
        delta_TT_TC_m_l0,
        delta_DIST_2R_m_l0,

        ASC_2R_w_l0, ASC_TC_w_l0,

        B_DIST_2R2_w_l0,
        B_DIST_WALK2_w_l0,
        B_TT_TC2_w_l0,
        delta_TT_TC_w_l0,
        delta_DIST_2R_w_l0,

        B_AGE_TC_m_l0, B_CADRE_TC_m_l0, B_AGE_2R_m_l0, B_CADRE_2R_m_l0,
        B_AGE_TC_w_l0, B_CADRE_TC_w_l0, B_AGE_2R_w_l0, B_CADRE_2R_w_l0,

    ) = params


    # ------- Utilités de l'homme
    # Le temps en transports en commun est plafonné à 3 heures, la distance à 10 km pour la marche
    # et à 25 km pour les deux-roues. Les coefficients linéaires s'écrivent -exp(delta) pour rester négatifs.
    XB_TC_m = B_AGE_TC_m_l0 * var('AGE40_m') + B_CADRE_TC_m_l0 * var('OCCP_PROFESSIONAL_m')

    XB_2R_m = B_AGE_2R_m_l0 * var('AGE40_m') + B_CADRE_2R_m_l0 * var('OCCP_PROFESSIONAL_m')

    V_TC_m = (ASC_TC_m_l0 + XB_TC_m
            - torch.exp(delta_TT_TC_m_l0) * torch.min(var('TT_TC_m'), torch.tensor((180/60)))
            + B_TT_TC2_m_l0 * (torch.min(var('TT_TC_m'), torch.tensor((180/60)))**2))

    V_WALK_m = (-3 * torch.min(var('DISTANCE_m'), torch.tensor(10))
              + B_DIST_WALK2_m_l0 * (torch.min(var('DISTANCE_m'), torch.tensor(10))**2)
              )

    V_2R_m = (ASC_2R_m_l0 + XB_2R_m
              - torch.exp(delta_DIST_2R_m_l0) * torch.min(var('DISTANCE_m'), torch.tensor(25))
              + B_DIST_2R2_m_l0 * (torch.min(var('DISTANCE_m'), torch.tensor(25))**2)
              )

    # ------- Utilités de la femme
    XB_TC_w = B_AGE_TC_w_l0 * var('AGE40_w') + B_CADRE_TC_w_l0 * var('OCCP_PROFESSIONAL_w')

    XB_2R_w = B_AGE_2R_w_l0 * var('AGE40_w') + B_CADRE_2R_w_l0 * var('OCCP_PROFESSIONAL_w')

    V_TC_w = (ASC_TC_w_l0 + XB_TC_w
            - torch.exp(delta_TT_TC_w_l0) * torch.min(var('TT_TC_w'), torch.tensor((180/60)))
            + B_TT_TC2_w_l0 * (torch.min(var('TT_TC_w'), torch.tensor((180/60)))**2))

    V_WALK_w = (-3 * torch.min(var('DISTANCE_w'), torch.tensor(10))
              + B_DIST_WALK2_w_l0 * (torch.min(var('DISTANCE_w'), torch.tensor(10))**2)
              )

    V_2R_w = (ASC_2R_w_l0 + XB_2R_w
              - torch.exp(delta_DIST_2R_w_l0) * torch.min(var('DISTANCE_w'), torch.tensor(25))
              + B_DIST_2R2_w_l0 * (torch.min(var('DISTANCE_w'), torch.tensor(25))**2)
              )

    # ------- Probabilités de l'homme
    # On retranche le maximum avant l'exponentielle pour éviter les dépassements numériques.
    V_men_stack = torch.stack([V_TC_m[idx_men] / sigma_l0,
                            V_WALK_m[idx_men] / sigma_l0,
                            V_2R_m[idx_men] / sigma_l0], dim=1)
    max_V_men = V_men_stack.max()
    exp_V_men = torch.exp(V_men_stack - max_V_men)
    sum_exp_men = exp_V_men.sum(dim=1)

    P_TC_m, P_WALK_m, P_2R_m = exp_V_men.T / sum_exp_men

    # ------- Probabilités de la femme
    V_women_stack = torch.stack([V_TC_w[idx_women] / sigma_l0,
                            V_WALK_w[idx_women] / sigma_l0,
                            V_2R_w[idx_women] / sigma_l0], dim=1)
    max_V_women = V_women_stack.max()
    exp_V_women = torch.exp(V_women_stack - max_V_women)
    sum_exp_women = exp_V_women.sum(dim=1)

    P_TC_w, P_WALK_w, P_2R_w = exp_V_women.T / sum_exp_women

    # ------- Choix observés

    choices = df['df']

    Choice_TC_m = torch.tensor((choices['COMMUTE_MODE_m']=='PUBLIC').astype(int).values, dtype=torch.float64)
    Choice_WALK_m = torch.tensor((choices['COMMUTE_MODE_m']=='WALK').astype(int).values, dtype=torch.float64)
    Choice_2R_m = torch.tensor((choices['COMMUTE_MODE_m']=='TWO_WHEELS').astype(int).values, dtype=torch.float64)

    Choice_TC_w = torch.tensor((choices['COMMUTE_MODE_w']=='PUBLIC').astype(int).values, dtype=torch.float64)
    Choice_WALK_w = torch.tensor((choices['COMMUTE_MODE_w']=='WALK').astype(int).values, dtype=torch.float64)
    Choice_2R_w = torch.tensor((choices['COMMUTE_MODE_w']=='TWO_WHEELS').astype(int).values, dtype=torch.float64)

    weights_m = var('WEIGHT_m')
    weights_m = weights_m[idx_men]
    weights_w = var('WEIGHT_w')
    weights_w = weights_w[idx_women]

    # Vraisemblance d'un modèle sans variable, qui attribue à chacun les parts observées
    if null_loglik:

        null_LL = torch.sum(weights_m * Choice_TC_m[idx_men] * torch.log(torch.sum(Choice_TC_m[idx_men])/len(idx_men))) + \
            torch.sum(weights_m * Choice_2R_m[idx_men] * torch.log(torch.sum(Choice_2R_m[idx_men])/len(idx_men))) + \
            torch.sum(weights_m * Choice_WALK_m[idx_men] * torch.log(torch.sum(Choice_WALK_m[idx_men])/len(idx_men))) + \
            torch.sum(weights_w * Choice_TC_w[idx_women] * torch.log(torch.sum(Choice_TC_w[idx_women])/len(idx_women))) + \
            torch.sum(weights_w * Choice_2R_w[idx_women] * torch.log(torch.sum(Choice_2R_w[idx_women])/len(idx_women))) + \
            torch.sum(weights_w * Choice_WALK_w[idx_women] * torch.log(torch.sum(Choice_WALK_w[idx_women])/len(idx_women)))
        return -null_LL

    epsilon = 1e-30


    LL = torch.sum(weights_m * Choice_TC_m[idx_men] * torch.log(torch.clamp(P_TC_m, min=epsilon))) + \
        torch.sum(weights_m * Choice_2R_m[idx_men] * torch.log(torch.clamp(P_2R_m, min=epsilon))) + \
        torch.sum(weights_m * Choice_WALK_m[idx_men] * torch.log(torch.clamp(P_WALK_m, min=epsilon))) + \
        torch.sum(weights_w * Choice_TC_w[idx_women] * torch.log(torch.clamp(P_TC_w, min=epsilon))) + \
        torch.sum(weights_w * Choice_2R_w[idx_women] * torch.log(torch.clamp(P_2R_w, min=epsilon))) + \
        torch.sum(weights_w * Choice_WALK_w[idx_women] * torch.log(torch.clamp(P_WALK_w, min=epsilon)))

    if pytorch:
        return -LL

    # Utilité espérée des modes sans voiture, transmise au niveau 1
    if logsum:
        LS_m = sigma_l0 * torch.log(
            torch.exp(V_TC_m / sigma_l0)
            + torch.exp(V_WALK_m / sigma_l0) + torch.exp(V_2R_m / sigma_l0)
        )
        LS_w = sigma_l0 * torch.log(
            torch.exp(V_TC_w / sigma_l0)
            + torch.exp(V_WALK_w / sigma_l0) + torch.exp(V_2R_w / sigma_l0)
        )
        return LS_m, LS_w

    return (-LL).detach().numpy()


def level1(params, df, pytorch=False, grad=False, null_loglik=False,
           pareto_weight=False, logsum=False, utility_nestB=False):
    """Niveau 1 : qui prend la voiture dans le couple, à nombre de voitures donné.

    Notations pour chaque conjoint : Ca = seul en voiture, Cd = conduit l'autre,
    Cp = passager, B = sans voiture (son utilité vient du niveau 0).
    L'utilité du couple est une moyenne des utilités des deux conjoints, pondérée
    par Lambda (poids de la femme) et 1 - Lambda (poids de l'homme).
    Avec une seule voiture, l'option "chacun la sienne" (CaCa) n'existe pas.
    """
    v = df["vars"]
    idx1 = df["is_one_car_idx"]
    idx2 = df["is_multi_car_idx"]

    def var(name): return v[name]

    if grad:
        params = params.clone().requires_grad_(True)
    else:
        params = torch.tensor(params, dtype=torch.float64)

    (
    sigma_1c_l1, sigma_2c_l1,

    ASC_Cp_m_l1, ASC_Cp_w_l1,
    ASC_Cd_m_l1, ASC_Cd_w_l1,

    ASC_B_w_l1, delta_Ca_w_l1, delta_Cd_w_l1, delta_Cp_w_l1,
    ASC_B_m_l1, delta_Ca_m_l1, delta_Cp_m_l1, delta_Cd_m_l1,

    L_FEMME_PLUS_DIPL_l1, L_HOMME_PLUS_DIPL_l1,
    B_ENF_B_m_l1, B_ENF_B_w_l1,

    ) = params

    # Poids de la femme dans les choix du couple : 0,5 à diplôme égal
    Xb_Lambda = L_FEMME_PLUS_DIPL_l1 * var('WOMANmoreEDU') + L_HOMME_PLUS_DIPL_l1 * var('MANmoreEDU')

    Lambda = 1 / (1 + torch.exp(-Xb_Lambda))

    # Un enfant de moins de 6 ans modifie l'attrait de l'option sans voiture
    Xb_m_OTHERS = B_ENF_B_m_l1 * var('ENF_0_5')
    Xb_w_OTHERS = B_ENF_B_w_l1 * var('ENF_0_5')

    # ------- Utilités individuelles
    # Le coût du temps en voiture s'écrit exp(delta) pour rester positif. Le conducteur
    # supporte en plus le détour (WOMANtowardsMAN, MANtowardsWOMAN) pour déposer son conjoint.
    VOT_Ca_m = torch.exp(delta_Ca_m_l1)

    VOT_Ca_w = torch.exp(delta_Ca_w_l1)

    V_Ca_m = -VOT_Ca_m*var('TT_VP_m')
    V_Ca_w = -VOT_Ca_w*var('TT_VP_w')

    V_Cp_m = ASC_Cp_m_l1 - VOT_Ca_m*torch.exp(delta_Cp_m_l1)*var('TT_VP_m')
    V_Cp_w = ASC_Cp_w_l1 - VOT_Ca_w*torch.exp(delta_Cp_w_l1)*var('TT_VP_w')

    V_Cd_m = ASC_Cd_m_l1 - VOT_Ca_m*torch.exp(delta_Cd_m_l1)*var('TT_VP_w') - VOT_Ca_m*var('WOMANtowardsMAN')
    V_Cd_w = ASC_Cd_w_l1 - VOT_Ca_w*torch.exp(delta_Cd_w_l1)*var('TT_VP_m') - VOT_Ca_w*var('MANtowardsWOMAN')

    V_B_m = (ASC_B_m_l1 + Xb_m_OTHERS) + var('LS_m')
    V_B_w = (ASC_B_w_l1 + Xb_w_OTHERS) + var('LS_w')

    # ------- Utilités du couple (premier terme : elle, second terme : lui)
    V_CaCa = Lambda * V_Ca_w + (1 - Lambda) * V_Ca_m
    V_BB = Lambda * V_B_w + (1 - Lambda) * V_B_m
    V_CaB = Lambda * V_Ca_w + (1 - Lambda) * V_B_m
    V_BCa = Lambda * V_B_w + (1 - Lambda) * V_Ca_m
    V_CdCp = Lambda * V_Cd_w + (1 - Lambda) * V_Cp_m
    V_CpCd = Lambda * V_Cp_w + (1 - Lambda) * V_Cd_m

    # ------- Probabilités, couples à une voiture
    V_1c_stack = torch.stack([
        V_BB[idx1] / sigma_1c_l1,
        V_CaB[idx1] / sigma_1c_l1,
        V_BCa[idx1] / sigma_1c_l1,
        V_CdCp[idx1] / sigma_1c_l1,
        V_CpCd[idx1] / sigma_1c_l1
    ], dim=1)
    max_V_1c = V_1c_stack.max()
    exp_V_1c = torch.exp(V_1c_stack - max_V_1c)
    sum_exp_1c = exp_V_1c.sum(dim=1)

    P_BB_1c, P_CaB_1c, P_BCa_1c, P_CdCp_1c, P_CpCd_1c = exp_V_1c.T / sum_exp_1c

    # ------- Probabilités, couples à deux voitures ou plus
    V_2c_stack = torch.stack([
        V_CaCa[idx2] / sigma_2c_l1,
        V_BB[idx2] / sigma_2c_l1,
        V_CaB[idx2] / sigma_2c_l1,
        V_BCa[idx2] / sigma_2c_l1,
        V_CdCp[idx2] / sigma_2c_l1,
        V_CpCd[idx2] / sigma_2c_l1
    ], dim=1)
    max_V_2c = V_2c_stack.max()
    exp_V_2c = torch.exp(V_2c_stack - max_V_2c)
    sum_exp_2c = exp_V_2c.sum(dim=1)

    P_CaCa_2c, P_BB_2c, P_CaB_2c, P_BCa_2c, P_CdCp_2c, P_CpCd_2c = exp_V_2c.T / sum_exp_2c

    # ------- Choix observés
    # Code TRANS du recensement 2015 : 4 = voiture ; 2, 3 et 5 = marche, deux-roues, transports en commun.
    # Quand les deux conjoints vont travailler en voiture, on ne sait pas s'ils partagent le véhicule.
    choices = df['df']
    en_voiture_w, en_voiture_m = choices['TRANS_w'] == 4, choices['TRANS_m'] == 4
    autre_w, autre_m = choices['TRANS_w'].isin([2, 3, 5]), choices['TRANS_m'].isin([2, 3, 5])
    Choice_CC = torch.tensor((en_voiture_w & en_voiture_m).to_numpy().astype(float))
    Choice_BB = torch.tensor((autre_w & autre_m).to_numpy().astype(float))
    Choice_CaB = torch.tensor((en_voiture_w & autre_m).to_numpy().astype(float))
    Choice_BCa = torch.tensor((autre_w & en_voiture_m).to_numpy().astype(float))

    w_1c = var('WEIGHT_hh')
    w_1c = w_1c[idx1]
    w_2c = var('WEIGHT_hh')
    w_2c = w_2c[idx2]

    epsilon = 1e-30

    LL_1c = (
    torch.sum(w_1c * Choice_CC[idx1] * torch.log(torch.clamp(P_CdCp_1c + P_CpCd_1c, min=epsilon))) +
    torch.sum(w_1c * Choice_BB[idx1] * torch.log(torch.clamp(P_BB_1c, min=epsilon))) +
    torch.sum(w_1c * Choice_CaB[idx1] * torch.log(torch.clamp(P_CaB_1c, min=epsilon))) +
    torch.sum(w_1c * Choice_BCa[idx1] * torch.log(torch.clamp(P_BCa_1c, min=epsilon)))
    )

    LL_2c = (
    torch.sum(w_2c * Choice_CC[idx2] * torch.log(torch.clamp(P_CaCa_2c + P_CdCp_2c + P_CpCd_2c, min=epsilon))) +
    torch.sum(w_2c * Choice_BB[idx2] * torch.log(torch.clamp(P_BB_2c, min=epsilon))) +
    torch.sum(w_2c * Choice_CaB[idx2] * torch.log(torch.clamp(P_CaB_2c, min=epsilon))) +
    torch.sum(w_2c * Choice_BCa[idx2] * torch.log(torch.clamp(P_BCa_2c, min=epsilon)))
    )

    if pytorch:
        return -(LL_1c + LL_2c)
    # Vraisemblance d'un modèle sans variable, qui attribue à chacun les parts observées
    if null_loglik:
        n_1c = len(idx1)
        n_2c = len(idx2)

        Choice_CC_1c, Choice_BB_1c, Choice_CaB_1c, Choice_BCa_1c = (
        Choice_CC[idx1], Choice_BB[idx1], Choice_CaB[idx1], Choice_BCa[idx1])
        Choice_CC_2c, Choice_BB_2c, Choice_CaB_2c, Choice_BCa_2c = (
        Choice_CC[idx2], Choice_BB[idx2], Choice_CaB[idx2], Choice_BCa[idx2])

        # parts observées
        share_CC_1c = torch.sum(Choice_CC_1c) / n_1c
        share_BB_1c = torch.sum(Choice_BB_1c) / n_1c
        share_CaB_1c = torch.sum(Choice_CaB_1c) / n_1c
        share_BCa_1c = torch.sum(Choice_BCa_1c) / n_1c

        share_CC_2c = torch.sum(Choice_CC_2c) / n_2c
        share_BB_2c = torch.sum(Choice_BB_2c) / n_2c
        share_CaB_2c = torch.sum(Choice_CaB_2c) / n_2c
        share_BCa_2c = torch.sum(Choice_BCa_2c) / n_2c

        null_LL_1c = (
        torch.sum(w_1c * Choice_CC_1c * torch.log(torch.clamp(share_CC_1c, min=epsilon))) +
        torch.sum(w_1c * Choice_BB_1c * torch.log(torch.clamp(share_BB_1c, min=epsilon))) +
        torch.sum(w_1c * Choice_CaB_1c * torch.log(torch.clamp(share_CaB_1c, min=epsilon))) +
        torch.sum(w_1c * Choice_BCa_1c * torch.log(torch.clamp(share_BCa_1c, min=epsilon)))
        )

        null_LL_2c = (
        torch.sum(w_2c * Choice_CC_2c * torch.log(torch.clamp(share_CC_2c, min=epsilon))) +
        torch.sum(w_2c * Choice_BB_2c * torch.log(torch.clamp(share_BB_2c, min=epsilon))) +
        torch.sum(w_2c * Choice_CaB_2c * torch.log(torch.clamp(share_CaB_2c, min=epsilon))) +
        torch.sum(w_2c * Choice_BCa_2c * torch.log(torch.clamp(share_BCa_2c, min=epsilon)))
        )

        return -(null_LL_1c + null_LL_2c)

    if pareto_weight:
        return Lambda * torch.ones_like(var('VOIT'))

    # Utilité espérée du partage de la voiture, transmise au niveau 2
    if logsum:
        LS_1c = sigma_1c_l1 * torch.log(
            torch.exp(V_BB/sigma_1c_l1) + torch.exp(V_CaB/sigma_1c_l1)
            + torch.exp(V_BCa/sigma_1c_l1) + torch.exp(V_CpCd/sigma_1c_l1) + torch.exp(V_CdCp/sigma_1c_l1)
        )
        LS_2c = sigma_2c_l1 * torch.log(
            torch.exp(V_CaCa/sigma_2c_l1) + torch.exp(V_BB/sigma_2c_l1) + torch.exp(V_CaB/sigma_2c_l1)
            + torch.exp(V_BCa/sigma_2c_l1) + torch.exp(V_CpCd/sigma_2c_l1) + torch.exp(V_CdCp/sigma_2c_l1)
        )
        return (LS_1c).detach().numpy(), (LS_2c).detach().numpy()
    if utility_nestB:
        return V_BB.detach().numpy()

    return -(LL_1c + LL_2c).detach().numpy()


def level2(params, df, pytorch=False, grad=False, null_loglik=False, logsum=False):
    """Niveau 2 : une voiture ou deux, pour les couples qui en ont au moins une."""
    v = df["vars"]
    idx = df["at_least_one_car_idx"]

    def var(name): return v[name]

    if grad:
        params = params.clone().requires_grad_(True)
    else:
        params = torch.tensor(params, dtype=torch.float64)

    (sigma_l2,

    ASC_2c_l2,

    B_ENF_l2, B_PROPRIO_l2, B_PARKING_l2, B_GC_l2) = params

    XB_2c = (B_ENF_l2 * var('ENF_0_16') + B_PROPRIO_l2 * var('HOMEOWNERSHIP')
        + B_PARKING_l2 * var('PARKING') + B_GC_l2 * var('RES_OUTERRING'))

    V_1car = var('LS_1c')
    V_2car = (ASC_2c_l2 + XB_2c + var('LS_2c'))

    # ------- Probabilités
    V_stack = torch.stack([
        V_1car[idx] / sigma_l2,
        V_2car[idx] / sigma_l2
    ], dim=1)
    max_V = V_stack.max()
    exp_V = torch.exp(V_stack - max_V)
    sum_exp = exp_V.sum(dim=1)

    P_1car, P_2car = exp_V.T / sum_exp

    choice = df['df']
    CHOICE_1CAR = torch.tensor((choice['VOIT'] == 1).astype(int), dtype=torch.float64)
    CHOICE_2CAR = torch.tensor((choice['VOIT'] >= 2).astype(int), dtype=torch.float64)

    w = var('WEIGHT_hh')
    w = w[idx]

    epsilon=1e-30
    LL = (torch.sum(w * CHOICE_1CAR[idx] * torch.log(torch.clamp(P_1car, min=epsilon))) +
     torch.sum(w * CHOICE_2CAR[idx] * torch.log(torch.clamp(P_2car, min=epsilon))))

    if pytorch:
        return -LL

    if null_loglik:
        null_LL = (torch.sum(w * CHOICE_1CAR[idx] * torch.log(torch.sum(CHOICE_1CAR[idx])/len(idx)))
            + torch.sum(w * CHOICE_2CAR[idx] * torch.log(torch.sum(CHOICE_2CAR[idx])/len(idx))))
        return -null_LL

    # Utilité espérée d'avoir une voiture ou plus, transmise au niveau 3
    if logsum:
        LS_level2 = sigma_l2 * torch.log(
            torch.exp(V_1car/sigma_l2) + torch.exp(V_2car/sigma_l2)
        )
        return (LS_level2).detach().numpy()

    return (-LL).detach().numpy()


def level3(params, df, pytorch=False, grad=False, null_loglik=False, logsum=False):
    """Niveau 3 : avoir au moins une voiture, ou aucune (tous les couples)."""
    v = df["vars"]

    def var(name): return v[name]

    if grad:
        params = params.clone().requires_grad_(True)
    else:
        params = torch.tensor(params, dtype=torch.float64)

    (sigma_l3,

    ASC_car_l3,

    B_ENF_l3, B_PROPRIO_l3, B_PARKING_l3, B_GC_l3) = params

    XB_car = (B_ENF_l3 * var('ENF_0_16') + B_PROPRIO_l3 * var('HOMEOWNERSHIP')
         + B_PARKING_l3 * var('PARKING') + B_GC_l3 * var('RES_OUTERRING'))

    V_NoCar = var('V_BB')
    V_Car = ASC_car_l3 + XB_car + var('LS_car')

    V_stack = torch.stack([
        V_NoCar/sigma_l3,
        V_Car/sigma_l3], dim=1)
    max_V = V_stack.max()
    exp_V = torch.exp(V_stack - max_V)
    sum_exp = exp_V.sum(dim=1)

    P_NoCar, P_Car = exp_V.T / sum_exp

    choice = df['df']
    CHOICE_NOCAR = torch.tensor((choice['VOIT'] == 0).astype(int), dtype=torch.float64)
    CHOICE_CAR = torch.tensor((choice['VOIT'] >= 1).astype(int), dtype=torch.float64)

    epsilon=1e-30
    w = var('WEIGHT_hh')

    LL = (torch.sum(w * CHOICE_NOCAR * torch.log(torch.clamp(P_NoCar, min=epsilon))) +
        torch.sum(w * CHOICE_CAR * torch.log(torch.clamp(P_Car, min=epsilon))))

    if pytorch:
        return -LL

    if null_loglik:
        null_LL = (torch.sum(w * CHOICE_NOCAR * torch.log(torch.sum(CHOICE_NOCAR)/len(CHOICE_NOCAR)))
            + torch.sum(w * CHOICE_CAR * torch.log(torch.sum(CHOICE_CAR)/len(CHOICE_CAR))))
        return -null_LL

    if logsum:
        LS_level3 = sigma_l3 * torch.log(
        torch.exp(V_NoCar/sigma_l3) + torch.exp(V_Car/sigma_l3)
        )
        return (LS_level3).detach().numpy()

    return (-LL).detach().numpy()
