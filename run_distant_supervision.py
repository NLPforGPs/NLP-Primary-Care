"""
This script runs the distant supervision experiments with coarse-grained topics.
First, it runs cross validation with different stopwords selected for each shallow classifier and saves the results.
The first results are saved to distant_stopwords.csv.

Second, it runs distant supervision with ICPC-2 codes and computes results with and without class A: General.
Third, it runs distant supervision with CKS codes.
Fourth, it runs distant supervision with ICPC-2 codes excluding the patient's speech.

All of these results are saved to distant_descriptions.csv and includes cross validation and test set prec, rec and F1.

Fifth, we run all methods with the best descriptions on dev and test sets. We save this to distant_test.csv.
Sixth, we take the best overall method and save the results for each class to distant_per_class.csv.
"""
import numpy as np
import pandas as pd
from utils.metrics.metric import evaluate_classifications, evaluate_per_class
from shallow_classifiers import run_binary_naive_bayes, run_multiclass_naive_bayes, run_binary_svm, \
    run_multiclass_svm, run_nearest_centroid, run_nearest_neighbors
from prepare_data import prepare_original_data
from utils.utils import stratified_multi_label_split
from prepare_data import load_descriptions
from utils.stopwords import get_medical_stopwords, get_custom_stopwords, get_english_stopwords
from sklearn.feature_extraction.text import TfidfVectorizer


def set_stopwords(use_med, use_cus, use_eng):
    stopwords = []
    if use_med:
        stopwords += medical_stopwords
    if use_cus:
        stopwords += custom_stopwords
    if use_eng:
        stopwords += english_stopwords

    return stopwords


def get_description_vectors(description_corpus, stopwords):
    max_features = 5000

    text_vectorizer = TfidfVectorizer(ngram_range=(1,2), stop_words=stopwords, max_features=max_features)
    description_vec = text_vectorizer.fit_transform(description_corpus)
    print(f"icpc description bag-of-word matrix shape: {description_vec.shape}")
    vec_vocab = text_vectorizer.vocabulary_ # dictionary that contain the BOW tokens

    # print(f"bag-of-word tokens: {', '.join(list(vec_vocab.keys())[:5])}...")
    print(f"vocabulary size: {len(vec_vocab)}")

    return description_vec, text_vectorizer


def run_distant_supervision(method, description_corpus, y_desc, stopword_setting, dev_data, y_dev,
                            without_A_class=False, model=None):
    stopwords = set_stopwords('m' in stopword_setting, 'c' in stopword_setting, 'e' in stopword_setting)

    if 'BERT' not in method:
        description_vec, text_vectorizer = get_description_vectors(description_corpus, stopwords)
        X_dev = text_vectorizer.transform(dev_data[key])

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

        y_pred_mat_dev, _ = clf(description_vec, y_desc, X_dev)
        model = None

    results = evaluate_classifications(y_dev, y_pred_mat_dev, mult_lbl_enc.classes_, show_report=False)

    if without_A_class:
        # get the data without the A class examples:
        # A_class_idx = np.argwhere(mult_lbl_enc.classes_ == 'A')[0][0]  # A class is index 0
        results_without_A = evaluate_classifications(y_dev[:, 1:], y_pred_mat_dev[:, 1:], mult_lbl_enc.classes_[1:], show_report=False)
        return results[0], results[1], results[2], results_without_A[0], results_without_A[1], results_without_A[2], y_pred_mat_dev, model
    else:
        return results[0], results[1], results[2], y_pred_mat_dev, model


def run_stopword_experiment(methods, description_settings, stopword_settings, dev_data, y_dev):

    f1_dev = np.zeros((len(description_settings)*len(methods), len(stopword_settings)))
    row_labels = []

    for j, selected_mode in enumerate(description_settings):
        description_corpus = load_descriptions(selected_mode, mult_lbl_enc.classes_)
        for m, method in enumerate(methods):
            row_labels.append(f'{method}, {selected_mode}')

            for i, s in enumerate(stopword_settings):
                f1_dev[m + (j * len(methods)), i], _, _, _, _ = run_distant_supervision(method, description_corpus,
                                                                                        y_desc, s, dev_data, y_dev)

    df = pd.DataFrame(np.around(f1_dev, 3), index=row_labels, columns=stopword_settings, )
    df.to_csv(stopwords_file, sep=',')
    print(f'Saving results to {stopwords_file}')


if __name__ == '__main__':
    # Load the data
    # YOU MAY NEED TO SET THE LOCATION OF THE DATASET IN oneinamillion/resources.py.
    orig_dataset, mult_lbl_enc, y_hot = prepare_original_data()

    # Load our standard dev/test split
    dev_data, test_data, y_hot_dev, y_hot_test = stratified_multi_label_split(orig_dataset, y_hot)

    # Load our stopword lists for the shallow classifiers
    medical_stopwords = get_medical_stopwords()
    custom_stopwords = get_custom_stopwords()
    english_stopwords = get_english_stopwords()

    # Training labels for distant supervision. Each class is represented by one data point.
    y_desc = mult_lbl_enc.fit_transform(mult_lbl_enc.classes_)

    # EXPERIMENT 1 -- Stopwords. ----------------------------

    # Select which part of the transcripts we will use
    key = 'transcript__conversation_both'
    # key = 'transcript__conversation_gp'
    # key = 'transcript__conversation_patient'

    # Specify which descriptions we will test
    selected_modes = ['ICPC only', 'CKS only']

    stopword_settings = [
        [],
        'e',
        'm',
        'c',
        'mc',
        'ce',
        'mce'
    ]
    methods = [
        'multiclass NB'
    ]

    stopwords_file = 'results/distant_stopwords.csv'
    run_stopword_experiment(methods, selected_modes, stopword_settings, dev_data, y_hot_dev)

    # stopwords_file = 'results/distant_stopwords_test.csv'
    # run_distant_supervision(methods, selected_modes, stopword_settings, test_data, y_hot_test)

    # EXPERIMENT 2 & 3 -- Comparing key methods with ICPC versus CKS, and without patient's speech  -------------------
    # from the experiment one, the optimal setting is: ce for ICPC-2 and mce for CKS

    methods_for_description_test = [
        'multiclass NB',
        # 'BERT MLM'
    ]

    methods_for_best_descriptions = [
        'binary NB',
        'multiclass NB',
        'nearest centroid',
        # 'BERT classifier',
        # 'BERT MLM',
    ]

    f1 = np.zeros((len(methods), len(selected_modes)*2 + 1))
    prec = np.zeros((len(methods), len(selected_modes)*2 + 1))
    rec = np.zeros((len(methods), len(selected_modes)*2 + 1))

    csv_header = []

    # cache some predictions and models for later
    preds_dev = {}  # first key is description, second is method
    preds_test = {}
    models = {}
    for mode in selected_modes:
        preds_dev[mode] = {}
        preds_test[mode] = {}
        models[mode] = {}

    # run selected methods with complete transcripts, including A class and without it, with both sets of descriptions
    for d, mode in enumerate(selected_modes):
        description_corpus = load_descriptions(mode, mult_lbl_enc.classes_)
        stopword_setting = 'ce' if mode == 'ICPC only' else 'mce'
        for m, method in enumerate(methods_for_description_test):
            f1[m, d*2 + (d > 0)], _, _, f1[m, d*2 + 1 + (d > 0)], _, _, preds_dev[mode][method], models[mode][method] = \
                run_distant_supervision(method, description_corpus, y_desc, stopword_setting, dev_data, y_hot_dev,
                                        without_A_class=True)
            csv_header += [f'{mode}, {mode} without A, ']

    # EXPERIMENT 4 -- without patient's speech ---------------------------
    # run selected methods with ICPC codes only, without patients' speech
    key = 'transcript__conversation_gp'
    description_corpus = load_descriptions('ICPC only', mult_lbl_enc.classes_)

    for m, method in enumerate(methods_for_description_test):
        f1[m, 2], _, _, _, _ = run_distant_supervision(method, description_corpus, y_desc, 'ce', dev_data, y_hot_dev)
    csv_header = csv_header[0] + 'ICPC only, GP speech only' + ','.join(csv_header[1:])

    descriptions_file = './results/distant_descriptions.csv'
    np.savetxt(descriptions_file, f1, delimiter=',', fmt='%.3f', header=csv_header)

    # EXPERIMENT 5 -- run all methods with chosen descriptions and stopword setting -----------------------

    # run a larger list of methods with ICPC codes only, with both sets of speech -- the best setup overall
    key = 'transcript__conversation_both'
    csv_header = 'F1 (dev), prec (dev), rec (dev), F1 (test), prec (test), rec (test)'
    results = np.zeros((len(methods_for_best_descriptions), 6))
    mode = 'ICPC only'
    description_corpus = load_descriptions(mode, mult_lbl_enc.classes_)
    for m, method in enumerate(methods_for_best_descriptions):
        if method not in preds_dev[mode]:
            results[m, 0], results[m, 1], results[m, 2], preds_dev[mode][method], models[mode][method] = \
                run_distant_supervision(method, description_corpus, y_desc, 'ce', dev_data, y_hot_dev)
        else:  # don't rerun the model, just load the predictions and compute results
            results[m, 0], results[m, 1], results[m, 2] = evaluate_classifications(y_hot_dev, preds_dev[mode][method],
                                                                                   mult_lbl_enc.classes_,
                                                                                   show_report=False)

        # repeat on test set
        results[m, 3], results[m, 4], results[m, 5], preds_test[mode][method], _ = run_distant_supervision(
            method, description_corpus, y_desc, 'ce', test_data, y_hot_test, model=models[mode][method])

    test_file = './results/distant_test.csv'
    np.savetxt(test_file, results, delimiter=',', fmt='%.3f', header=csv_header)

    # EXPERIMENT 6 -- Save the per-class results for the best method -----------------------------------
    perclass_file = './results/distant_per_class.csv'
    selected_method = 'multiclass NB'

    dev_f1, dev_prec, dev_rec = evaluate_per_class(y_hot_dev, preds_dev[mode][selected_method])
    test_f1, test_prec, test_rec = evaluate_per_class(y_hot_test, preds_test[mode][selected_method])

    table = pd.DataFrame({
        'F1 (dev)': np.around(dev_f1, 3),
        'Prec (dev)': np.around(dev_prec, 3),
        'Rec (dev)': np.around(dev_rec, 3),
        'F1 (test)': np.around(test_f1, 3),
        'Prec (test)': np.around(test_prec, 3),
        'Rec (test)': np.around(test_rec, 3)
    }, index=mult_lbl_enc.classes_)

    table.to_csv(perclass_file, sep=',')