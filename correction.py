import numpy as np
from tqdm import tqdm
from sklearn.mixture import GaussianMixture as GMM


def label_correction(rel_overall, cd_overall, label_overall, args):

    pseudo_overall = []
    for j in range(label_overall.shape[1]):
        pseudo = label_correction_label_wise(rel_overall[:, j], cd_overall[:, j], label_overall[:, j], args)
        pseudo_overall.append(pseudo) 
    pseudo_overall = np.array(pseudo_overall).T
    return pseudo_overall

def label_correction_label_wise(rel_label_wise, cd_label_wise, label_label_wise, args):

    # get the hsm
    rel_label_wise = normalize_array(rel_label_wise)
    cd_label_wise = normalize_array(cd_label_wise)
    hsm_label_wise = rel_label_wise*args.alpha + cd_label_wise* (1-args.alpha)

    # get the positive and negative pairs
    pos_hsm_label_wise = hsm_label_wise[label_label_wise==1]
    neg_hsm_label_wise = hsm_label_wise[label_label_wise==0]

    # positive hsm
    pos_gmm = GMM(n_components=2, max_iter=100, tol=1e-2, reg_covar=5e-4)
    pos_gmm.fit(pos_hsm_label_wise.reshape)
    pos_prob = pos_gmm.predict_proba(pos_hsm_label_wise)
    pos_pseudo = positive_correction(pos_prob, args)

    # negative hsm
    neg_gmm = GMM(n_components=2, max_iter=100, tol=1e-2, reg_covar=5e-4)
    neg_gmm.fit(neg_hsm_label_wise)
    neg_prob = neg_gmm.predict_proba(neg_hsm_label_wise)
    neg_pseudo = negative_correction(neg_prob, args)

    # get pseudo label
    pseudo = []
    pos_idx = 0
    neg_idx = 0
    for label in label_label_wise:
        if label == 1:
            pseudo.append(pos_pseudo[pos_idx])
            pos_idx += 1
        else:
            pseudo.append(neg_pseudo[neg_idx])
            neg_idx += 1
    pseudo = np.array(pseudo)
    return pseudo

def positive_correction(pos_prob, args):
    pos_pseudo = []
    for prob in pos_prob:
        if prob[0] >0.5+args.epsilon:
            pseudo = 1
        elif prob[0] < 0.5-args.epsilon:
            pseudo = 0
        else:
            pseudo = prob[0]
        pos_pseudo.append(pseudo)
    pos_pseudo = np.array(pos_pseudo)
    return pos_pseudo

def negative_correction(neg_prob, args):
    neg_pseudo = []
    for prob in neg_prob:
        if prob[0] >0.5+args.epsilon:
            pseudo = 1
        elif prob[0] < 0.5-args.epsilon:
            pseudo = 0
        else:
            pseudo = prob[0]
        neg_pseudo.append(pseudo)
    neg_pseudo = np.array(neg_pseudo)
    return neg_pseudo


def normalize_array(rel_label_wise):
    rel_label_wise = np.array(rel_label_wise)
    rel_label_wise = (rel_label_wise - np.min(rel_label_wise)) / (np.max(rel_label_wise) - np.min(rel_label_wise))
    return rel_label_wise