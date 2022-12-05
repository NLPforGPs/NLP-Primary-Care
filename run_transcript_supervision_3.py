"""
This script runs the distant supervision experiments with coarse-grained topics.
First, it runs cross validation with different stopwords selected for each shallow classifier and saves the results.
The first results are saved to supervised_stopwords.csv.

Second, it runs transcript supervision on both dev and test sets and computes results with and without class A: General.
It saves the results for each class to supervised_test.csv.
"""
import numpy as np
import pandas as pd
from utils.metrics.metric import evaluate_classifications
from classifier_wrappers import run_binary_naive_bayes, run_multiclass_naive_bayes, run_binary_svm, \
    run_multiclass_svm, run_nearest_centroid, run_nearest_neighbors, run_bert_conventional, run_mlm_classifier, \
    run_nsp_classifier
from prepare_data import prepare_original_data
from utils.utils import stratified_multi_label_split
from utils.stopwords import get_medical_stopwords, get_custom_stopwords, get_english_stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold


key = 'transcript__conversation_both'
id2label = None


def set_stopwords(use_med, use_cus, use_eng):
    stopwords = []

    # Load our stopword lists for the shallow classifiers
    medical_stopwords = get_medical_stopwords()
    custom_stopwords = get_custom_stopwords()
    english_stopwords = get_english_stopwords()

    if use_med:
        stopwords += medical_stopwords
    if use_cus:
        stopwords += custom_stopwords
    if use_eng:
        stopwords += english_stopwords

    return stopwords


def test_single_split(train_set, y_train, test_set, y_test, stopwords, clf, preprocess, classes, k=0):
    if preprocess:
        max_features = 5000
        text_vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words=stopwords, max_features=max_features)
        X_train = text_vectorizer.fit_transform(train_set)
        X_test = text_vectorizer.transform(test_set)
    else:
        X_train = train_set
        X_test = test_set

    if preprocess:
        results = clf(X_train, y_train, X_test)  # performance on training set
    else:
        results = clf(X_train, y_train, X_test, k)  # performance on training set
    y_pred_mat = results[0]
    if len(results) == 3:
        model = results[2]
    else:
        model = None

    f1_k, p1_k, r1_k = evaluate_classifications(y_test, y_pred_mat, classes, show_report=False)
    f1_k_noA, p1_k_noA, r1_k_noA = evaluate_classifications(y_test[:, 1:], y_pred_mat[:, 1:], classes[1:],
                                                show_report=False)
    return f1_k, p1_k, r1_k, f1_k_noA, p1_k_noA, r1_k_noA, model


def cross_validate(clf, dev_data, y_dev, stopwords, seed, preprocess):

    f1s, p1s, r1s, f1s_noA, p1s_noA, r1s_noA = [], [], [], [], [], []

    kfolds = StratifiedKFold(n_splits=5, random_state=seed, shuffle=True)
    X_dev = dev_data[key].values

    k = 0
    for train_idxs, test_idxs in kfolds.split(X_dev, np.argmax(y_dev, 1)):
        train_set = X_dev[train_idxs]
        y_train = y_dev[train_idxs]

        test_set = X_dev[test_idxs]
        y_test = y_dev[test_idxs]

        f1_k, p1_k, r1_k, f1_k_noA, p1_k_noA, r1_k_noA, _ = test_single_split(train_set, y_train, test_set, y_test,
                                                                           stopwords, clf, preprocess,
                                                                           mult_lbl_enc.classes_, k)
        f1s.append(f1_k)
        p1s.append(p1_k)
        r1s.append(r1_k)
        f1s_noA.append(f1_k)
        p1s_noA.append(p1_k)
        r1s_noA.append(r1_k)

        k += 1

    f1 = np.mean(f1s)
    prec = np.mean(p1s)
    rec = np.mean(r1s)
    f1_noA = np.mean(f1s_noA)
    prec_noA = np.mean(p1s_noA)
    rec_noA = np.mean(r1s_noA)

    return f1, prec, rec, f1_noA, prec_noA, rec_noA


def run_transcript_supervision(method, stopword_setting, dev_data, y_dev, classes, test_data=None, y_test=None, seed=3, model=None):

    stopwords = set_stopwords('m' in stopword_setting, 'c' in stopword_setting, 'e' in stopword_setting)

    if 'BERT' in method:
        if method == 'BERT conventional':
            clf_unwrapped = run_bert_conventional
        elif method == 'BERT MLM':
            clf_unwrapped = run_mlm_classifier
        elif method == 'BERT NSP':
            clf_unwrapped = run_nsp_classifier

        trained_classifier = model

        def bert_clf(X_train, y_train, X_test, k):
            if test_data is None:
                run_name = 'cross_val_' + str(k)
            else:
                run_name = 'test'
            y_pred_mat, _, model = clf_unwrapped(X_train, y_train, id2label, X_test, run_name, 'supervised', trained_classifier=trained_classifier)
            return y_pred_mat, None

        global id2label
        if id2label is None:
            id2label = classes

        clf = bert_clf
    else:
        if method == 'nearest centroid':
            clf = run_nearest_centroid
        elif method == 'nearest neighbours':
            clf = run_nearest_neighbors
        elif method == 'binary NB':
            clf = run_binary_naive_bayes
        elif method == 'multiclass NB':
            clf = run_multiclass_naive_bayes
        elif method == 'binary SVM':
            clf = run_binary_svm
        elif method == 'multiclass SVM':
            clf = run_multiclass_svm

    if test_data is None:
        f1, prec, rec, f1_noA, prec_noA, rec_noA = cross_validate(clf, dev_data, y_dev, stopwords, seed,
                                                                  'BERT' not in method)
        model = None
    else:
        f1, prec, rec, f1_noA, prec_noA, rec_noA, model = test_single_split(dev_data[key].values, y_dev,
                                                                     test_data[key].values, y_test, stopwords, clf,
                                                                     'BERT' not in method, classes)

    return f1, prec, rec, f1_noA, prec_noA, rec_noA, model


def run_stopword_experiment(methods, stopword_settings, dev_data, y_dev):

    f1_dev = np.zeros((len(methods), len(stopword_settings)))

    for m, method in enumerate(methods):
        for i, s in enumerate(stopword_settings):
            f1_dev[m, i], _, _, _, _, _, _ = run_transcript_supervision(method, s, dev_data, y_dev, mult_lbl_enc.classes_,
                                                                     seed=3)
            print(f'Results for {method} with {s} stopwords. F1 = {f1_dev[m, i]}')

    # find best setting for each method:
    best_stopword_settings = np.argmax(f1_dev, axis=1)

    df = pd.DataFrame(np.around(f1_dev, 3), index=methods, columns=stopword_settings, )
    df.to_csv(stopwords_file, sep=',')
    print(f'Saving results to {stopwords_file}')

    return dict(zip(methods, stopword_settings[best_stopword_settings]))


if __name__ == '__main__':
    # Load the data
    # YOU MAY NEED TO SET THE LOCATION OF THE DATASET IN oneinamillion/resources.py.
    orig_dataset, mult_lbl_enc, y_hot = prepare_original_data()
    id2label = mult_lbl_enc.classes_

    # Load our standard dev/test split
    dev_data, test_data, y_hot_dev, y_hot_test = stratified_multi_label_split(orig_dataset, y_hot)

    # EXPERIMENT 1 -- Stopwords. ----------------------------

    # Select which part of the transcripts we will use
    key = 'transcript__conversation_both'
    # key = 'transcript__conversation_gp'
    # key = 'transcript__conversation_patient'

    stopword_settings = np.array([
        [],
        'e',
        'm',
        'c',
        'mc',
        'ce',
        'mce'
    ])
    methods = [
        'binary NB',
        'multiclass NB',
        'binary SVM',
        'multiclass SVM',
        'nearest centroid'
    ]

    X = np.arange(orig_dataset['index'].shape[0]).reshape((-1, 1))

    stopwords_file = 'results2/supervised_stopwords.csv'
    # best_stopword_settings = run_stopword_experiment(methods, stopword_settings, dev_data, y_hot_dev)
    best_stopword_settings = {}

    # EXPERIMENT 2 -- run all methods with chosen descriptions and stopword setting -----------------------

    methods = [
        'binary NB',
        # 'multiclass NB',
        # 'binary SVM',
        # 'multiclass SVM',
        # 'nearest centroid',
        # 'BERT MLM',
        # 'BERT NSP',
        'BERT conventional',
    ]
    # run a larger list of methods with ICPC codes only, with both sets of speech -- the best setup overall
    csv_header = 'F1 (dev), prec (dev), rec (dev), F1 (test), prec (test), rec (test)'
    results = np.zeros((len(methods), 6))

    trcsv_header = 'F1 (train), prec (train), rec (train)'
    trresults = np.zeros((len(methods), 3))

    stopwords_for_method = []
    for m, method in enumerate(methods):
        if method in best_stopword_settings:
            stopwords_for_method.append(best_stopword_settings[method])
        else:
            best_stopword_settings[method] = []
            stopwords_for_method.append([])

    for m, method in enumerate(methods):
        # test on dev set with cross validation
        results[m, 0], results[m, 1], results[m, 2], _, _, _, _ = run_transcript_supervision(
            method, best_stopword_settings[method], dev_data, y_hot_dev, mult_lbl_enc.classes_, seed=3)
        print(f'Results for {method} on dev set. F1 = {results[m, 0]}')

        # repeat on test set -- train on the whole dev set
        results[m, 3], results[m, 4], results[m, 5], _, _, _, model = run_transcript_supervision(
            method, best_stopword_settings[method], dev_data, y_hot_dev, mult_lbl_enc.classes_, test_data, y_hot_test)
        print(f'Results for {method} on test set. F1 = {results[m, 3]}')

        test_file = './results2/supervised_test3.csv'
        test_results = pd.DataFrame(np.around(results, 3), columns=csv_header.split(', '), index=methods)
        test_results['stopwords'] = stopwords_for_method
        test_results.to_csv(test_file, sep=',')

        # check for overfitting on the train set
        trresults[m, 0], trresults[m, 1], trresults[m, 2], _, _, _, _ = run_transcript_supervision(
            method, best_stopword_settings[method], dev_data, y_hot_dev, mult_lbl_enc.classes_, dev_data, y_hot_dev,
            seed=3, model=model)
        print(f'Results for {method} on train set. F1 = {results[m, 0]}')

        test_file = './results2/supervised_train3.csv'
        test_results = pd.DataFrame(np.around(trresults, 3), columns=trcsv_header.split(', '), index=methods)
        test_results.to_csv(test_file, sep=',')
