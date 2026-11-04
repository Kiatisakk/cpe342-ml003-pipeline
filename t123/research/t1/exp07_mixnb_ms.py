"""Mixture NB K=(6,3), 3 fold-seeds: supervised vs semi-supervised (unlabeled TRAIN rows in EM). OOFs saved."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from common import *
from exp04_mixnb import nb_matrix, cv_mixnb
lab, unl, te, ids = load(); y = lab[TARGET].astype(int).values
ref = pd.concat([lab, unl]); M = nb_matrix(lab, ref); Mu = nb_matrix(unl, ref)
a, o = multi_seed("exp07 mixnb K=(6,3) supervised", y, lambda s: cv_mixnb(M, y, s, K=(6, 3), smooth=1)[0]); np.save(f"{OUT}/exp07_mixnb.npy", np.array(o))
a, o = multi_seed("exp07 mixnb K=(6,3) semi-sup (+13k unlabeled train)", y, lambda s: cv_mixnb(M, y, s, K=(6, 3), smooth=1, Mu=Mu)[0]); np.save(f"{OUT}/exp07_mixnb_ss.npy", np.array(o))
