import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score


def compute_confidence_intervals(exptname, runnames, methodname):
    p_sample_list = []
    r_sample_list = []
    f_sample_list = []

    p_list = []
    r_list = []
    f_list = []

    print(f'Confidence intervals for {exptname}, {runnames}, {methodname}')

    for runname in runnames:
        filename = f'./output_predictions/{exptname}_{runname}_{methodname}'

        data = pd.read_csv(filename, index_col=0)
        # print(data.columns)
        # print(data.shape)
        nclasses = 16 if 'withouta' not in exptname else 15
        y_gold = data.to_numpy()[:, :nclasses].astype(int)
        y_pred = data.to_numpy()[:, nclasses:].astype(int)

        print(y_gold.shape)
        print(y_pred.shape)

        p_samples = []
        r_samples = []
        f_samples = []

        n_samples = 1000
        idxs = np.arange(y_gold.shape[0])  # list of all indexes to sample from
        for n in range(n_samples):
            # bootstrap sampling
            boostrap_idxs = np.random.choice(idxs, size=y_gold.shape[0])

            targets = y_gold[boostrap_idxs]
            predictions = y_pred[boostrap_idxs]

            keepclasses = np.any(targets, axis=0).astype(bool)
            targets = targets[:, keepclasses]
            predictions = predictions[:, keepclasses]

            p_n = precision_score(targets, predictions, average='macro', zero_division=0)
            r_n = recall_score(targets, predictions, average='macro', zero_division=0)
            f_n = f1_score(targets, predictions, average='macro', zero_division=0)

            p_samples.append(p_n)
            r_samples.append(r_n)
            f_samples.append(f_n)

        p_sample_list.append(p_samples)
        r_sample_list.append(r_samples)
        f_sample_list.append(f_samples)

        p = precision_score(y_gold, y_pred, average='macro', zero_division=0)
        r = recall_score(y_gold, y_pred, average='macro', zero_division=0)
        f = f1_score(y_gold, y_pred, average='macro', zero_division=0)

        p_list.append(p)
        r_list.append(r)
        f_list.append(f)


    p = np.mean(p_list)
    p_samples = np.mean(p_sample_list, axis=0)

    r = np.mean(r_list)
    r_samples = np.mean(r_sample_list, axis=0)

    f = np.mean(f_list)
    f_samples = np.mean(f_sample_list, axis=0)

    # print(np.around(p_samples, 2))
    p_lower = np.percentile(p_samples, 5)
    p_upper = np.percentile(p_samples, 95)
    print(f'Precision = {np.around(p, 3)}({np.around(p_lower, 3)},{np.around(p_upper, 3)})')
    # print(np.around(p_samples, 2))

    r_lower = np.percentile(r_samples, 5)
    r_upper = np.percentile(r_samples, 95)
    print(f'Recall = {np.around(r, 3)}({np.around(r_lower, 3)},{np.around(r_upper, 3)})')

    f_lower = np.percentile(f_samples, 5)
    f_upper = np.percentile(f_samples, 95)
    print(f'F1 = {np.around(f, 3)}({np.around(f_lower, 3)},{np.around(f_upper, 3)})')


# exptname = 'transcriptsupervision'
# runnames = [f'cross_val_{i}' for i in range(5)]
# methodname = 'binary NB'
# # compute_confidence_intervals(exptname, runnames, methodname)
# runnames = ['test']
# compute_confidence_intervals(exptname, runnames, methodname)
#
# exptname = 'transcriptsupervision'
# runnames = [f'cross_val_{i}' for i in range(5)]
# methodname = 'multiclass NB'
# compute_confidence_intervals(exptname, runnames, methodname)
# runnames = ['test']
# compute_confidence_intervals(exptname, runnames, methodname)
#
# exptname = 'transcriptsupervision'
# runnames = [f'cross_val_{i}' for i in range(5)]
# methodname = 'binary SVM'
# compute_confidence_intervals(exptname, runnames, methodname)
# runnames = ['test']
# compute_confidence_intervals(exptname, runnames, methodname)
#
# exptname = 'transcriptsupervision'
# runnames = [f'cross_val_{i}' for i in range(5)]
# methodname = 'multiclass SVM'
# compute_confidence_intervals(exptname, runnames, methodname)
# runnames = ['test']
# compute_confidence_intervals(exptname, runnames, methodname)
#
# exptname = 'transcriptsupervision'
# runnames = [f'cross_val_{i}' for i in range(5)]
# methodname = 'nearest centroid'
# compute_confidence_intervals(exptname, runnames, methodname)
# runnames = ['test']
# compute_confidence_intervals(exptname, runnames, methodname)

# ICPC
# exptname = 'distantsupervision_icpc_only'
# runnames = ['dev']
# methodname = 'binary NB'
# compute_confidence_intervals(exptname, runnames, methodname)
# runnames = ['test']
# compute_confidence_intervals(exptname, runnames, methodname)
#
# exptname = 'distantsupervision_icpc_only'
# runnames = ['dev']
# methodname = 'multiclass NB'
# compute_confidence_intervals(exptname, runnames, methodname)
# runnames = ['test']
# compute_confidence_intervals(exptname, runnames, methodname)
#
# exptname = 'distantsupervision_icpc_only'
# runnames = ['dev']
# methodname = 'nearest centroid'
# compute_confidence_intervals(exptname, runnames, methodname)
# runnames = ['test']
# compute_confidence_intervals(exptname, runnames, methodname)

# ICPC without A
exptname = 'distantsupervision_icpc_only_withouta'
runnames = ['dev']
methodname = 'binary NB'
compute_confidence_intervals(exptname, runnames, methodname)

exptname = 'distantsupervision_icpc_only_withouta'
runnames = ['dev']
methodname = 'multiclass NB'
compute_confidence_intervals(exptname, runnames, methodname)


exptname = 'distantsupervision_icpc_only_withouta'
runnames = ['dev']
methodname = 'nearest centroid'
compute_confidence_intervals(exptname, runnames, methodname)


# ICPC without patient's speech
exptname = 'distantsupervision_icpc_only_gponly'
runnames = ['dev']
methodname = 'binary NB'
compute_confidence_intervals(exptname, runnames, methodname)


exptname = 'distantsupervision_icpc_only_gponly'
runnames = ['dev']
methodname = 'multiclass NB'
compute_confidence_intervals(exptname, runnames, methodname)


exptname = 'distantsupervision_icpc_only_gponly'
runnames = ['dev']
methodname = 'nearest centroid'
compute_confidence_intervals(exptname, runnames, methodname)


#CKS
exptname = 'distantsupervision_cks_only'
runnames = ['dev']
methodname = 'binary NB'
compute_confidence_intervals(exptname, runnames, methodname)

exptname = 'distantsupervision_cks_only'
runnames = ['dev']
methodname = 'multiclass NB'
compute_confidence_intervals(exptname, runnames, methodname)

exptname = 'distantsupervision_cks_only'
runnames = ['dev']
methodname = 'nearest centroid'
compute_confidence_intervals(exptname, runnames, methodname)

# CKS without A
exptname = 'distantsupervision_cks_only_withouta'
runnames = ['dev']
methodname = 'binary NB'
compute_confidence_intervals(exptname, runnames, methodname)


exptname = 'distantsupervision_cks_only_withouta'
runnames = ['dev']
methodname = 'multiclass NB'
compute_confidence_intervals(exptname, runnames, methodname)


exptname = 'distantsupervision_cks_only_withouta'
runnames = ['dev']
methodname = 'nearest centroid'
compute_confidence_intervals(exptname, runnames, methodname)


# CKS without patient speech
exptname = 'distantsupervision_cks_only_gponly'
runnames = ['dev']
methodname = 'binary NB'
compute_confidence_intervals(exptname, runnames, methodname)


exptname = 'distantsupervision_cks_only_gponly'
runnames = ['dev']
methodname = 'multiclass NB'
compute_confidence_intervals(exptname, runnames, methodname)


exptname = 'distantsupervision_cks_only_gponly'
runnames = ['dev']
methodname = 'nearest centroid'
compute_confidence_intervals(exptname, runnames, methodname)


# Both
exptname = 'distantsupervision_both'
runnames = ['dev']
methodname = 'binary NB'
compute_confidence_intervals(exptname, runnames, methodname)

exptname = 'distantsupervision_both'
runnames = ['dev']
methodname = 'multiclass NB'
compute_confidence_intervals(exptname, runnames, methodname)

exptname = 'distantsupervision_both'
runnames = ['dev']
methodname = 'nearest centroid'
compute_confidence_intervals(exptname, runnames, methodname)

# Both without A
exptname = 'distantsupervision_both_withouta'
runnames = ['dev']
methodname = 'binary NB'
compute_confidence_intervals(exptname, runnames, methodname)


exptname = 'distantsupervision_both_withouta'
runnames = ['dev']
methodname = 'multiclass NB'
compute_confidence_intervals(exptname, runnames, methodname)


exptname = 'distantsupervision_both_withouta'
runnames = ['dev']
methodname = 'nearest centroid'
compute_confidence_intervals(exptname, runnames, methodname)


# Both without patient
exptname = 'distantsupervision_both_gponly'
runnames = ['dev']
methodname = 'binary NB'
compute_confidence_intervals(exptname, runnames, methodname)


exptname = 'distantsupervision_both_gponly'
runnames = ['dev']
methodname = 'multiclass NB'
compute_confidence_intervals(exptname, runnames, methodname)


exptname = 'distantsupervision_both_gponly'
runnames = ['dev']
methodname = 'nearest centroid'
compute_confidence_intervals(exptname, runnames, methodname)
