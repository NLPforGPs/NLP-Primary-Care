"""
This script runs the distant supervision experiments with coarse-grained topics.
First, it test validation set performance with different stopwords selected for each shallow classifier and saves the results.
The first results are saved to distant_stopwords.csv.

Second, it runs distant supervision with ICPC-2 codes compared to CKS codes,
 and computes results with and without class A: General, and
Third, it runs distant supervision with ICPC-2 codes excluding the patient's speech.

All of these results are saved to distant_descriptions.csv and includes cross validation and test set prec, rec and F1.

Fourth, we run all methods with the best descriptions on dev and test sets. We save this to distant_test.csv.
Fifth, we take the best overall method and save the results for each class to distant_per_class.csv.
"""
import argparse

import numpy as np
import pandas as pd
from utils.metrics.metric import evaluate_classifications, evaluate_per_class
from classifier_wrappers import run_binary_naive_bayes, run_multiclass_naive_bayes, run_binary_svm, \
    run_multiclass_svm, run_nearest_centroid, run_nearest_neighbors, run_bert_classifier, run_bert_conventional, \
    run_mlm_classifier, run_nsp_classifier
from prepare_data import prepare_original_data
from utils.utils import stratified_multi_label_split, save_predictions_to_file
from prepare_data import load_descriptions
from utils.stopwords import get_medical_stopwords, get_custom_stopwords, get_english_stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

save_predictions = True


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


def run_distant_supervision(method, run_name_template, description_corpus, y_desc, stopword_setting, dev_data, y_dev,
                            classes, existing_model=None):
    run_name = f"{run_name_template.lower().replace(' ', '_')}"

    if 'BERT' in method:

        if method == 'BERT conventional':
            clf = run_bert_conventional
        elif method == 'BERT MLM':
            clf = run_mlm_classifier
        elif method == 'BERT NSP':
            clf = run_nsp_classifier

        y_pred_mat_dev, _, model = clf(description_corpus, y_desc, classes, dev_data[key], run_name, run_name_template, existing_model)
    else:
        stopwords = set_stopwords('m' in stopword_setting, 'c' in stopword_setting, 'e' in stopword_setting)

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
        model = clf

    if save_predictions:
        if existing_model is not None:
            run_name = run_name + '_test'
        else:
            run_name = run_name + '_dev'
        save_predictions_to_file(y_dev, y_pred_mat_dev, f'distantsupervision_{run_name}_{method}')

    results = evaluate_classifications(y_dev, y_pred_mat_dev, classes, show_report=False)

    return results[0], results[1], results[2], y_pred_mat_dev, model


def run_stopword_experiment(methods, description_settings, stopword_settings, dev_data, y_dev):

    f1_dev = np.zeros((len(description_settings)*len(methods), len(stopword_settings)))
    row_labels = []

    for j, selected_mode in enumerate(description_settings):
        description_corpus = load_descriptions(selected_mode, mult_lbl_enc.classes_)
        for m, method in enumerate(methods):
            row_labels.append(f'{method}, {selected_mode}')

            for i, s in enumerate(stopword_settings):
                f1_dev[m + (j * len(methods)), i], _, _, _, _ = run_distant_supervision(method,
                                                                                        selected_mode + '_stopwords',
                                                                                        description_corpus,
                                                                                        y_desc, s, dev_data, y_dev,
                                                                                        mult_lbl_enc.classes_)

    df = pd.DataFrame(np.around(f1_dev, 3), index=row_labels, columns=stopword_settings, )
    df.to_csv(stopwords_file, sep=',')
    print(f'Saving results to {stopwords_file}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_to_run', type=str, default='1234')
    parser.add_argument('--methods', type=str, default=None)
    parser.add_argument('--mode', type=str, default=None)
    args = parser.parse_args()

    # Load the data
    # YOU MAY NEED TO SET THE LOCATION OF THE DATASET IN oneinamillion/resources.py.
    orig_dataset, mult_lbl_enc, y_hot = prepare_original_data()

    labels = {}
    for codes in orig_dataset['codes']:
        # print(codes)
        for code in codes:
            if code not in labels:
                labels[code] = 0

            labels[code] += 1

    # Load our standard dev/test split
    dev_data, test_data, y_hot_dev, y_hot_test = stratified_multi_label_split(orig_dataset, y_hot)

    # Load our stopword lists for the shallow classifiers
    medical_stopwords = get_medical_stopwords()
    custom_stopwords = get_custom_stopwords()
    english_stopwords = get_english_stopwords()

    # Training labels for distant supervision. Each class is represented by one data point.
    y_desc = mult_lbl_enc.fit_transform(mult_lbl_enc.classes_)
    id2label = mult_lbl_enc.classes_

    # EXPERIMENT 1 -- Stopwords. ----------------------------

    # Select which part of the transcripts we will use
    key = 'transcript__conversation_both'
    # key = 'transcript__conversation_gp'
    # key = 'transcript__conversation_patient'

    # Specify which descriptions we will test
    if args.mode is None:
        selected_modes = ['both', 'CKS only', 'ICPC only']
    else:
        selected_modes = [args.mode]

    stopword_settings = [
        [],
        'e',
        'm',
        'c',
        'mc',
        'ce',
        'mce'
    ]
    if args.methods is None:
        methods = [
            'binary NB',
            'multiclass NB',
            'nearest centroid',
            'BERT NSP',
            'BERT conventional',
            'BERT MLM',
        ]
    else:
        methods = args.methods.split(',')

    stopwords_file = 'results2/distant_stopwords.csv'
    if '1' in args.experiment_to_run:
        run_stopword_experiment(methods, selected_modes, stopword_settings, dev_data, y_hot_dev)

    # EXPERIMENT 2  -- Comparing key methods with ICPC versus CKS -------------------
    # from the experiment one, the optimal setting is: ce for ICPC-2 and mce for CKS

    csv_header = []
    for mode in selected_modes:
        csv_header += [f'{mode}', f'{mode} without A', f'{mode} GP speech only']
    ncols = len(csv_header)

    f1 = np.zeros((len(methods), ncols))
    prec = np.zeros((len(methods), ncols))
    rec = np.zeros((len(methods), ncols))

    # cache some predictions and models for later
    preds_dev = {}  # first key is description, second is method
    preds_test = {}
    models = {}
    for mode in selected_modes:
        preds_dev[mode] = {}
        preds_test[mode] = {}
        models[mode] = {}

    # run selected methods with complete transcripts, including A class and without it, with both sets of descriptions
    if '2' in args.experiment_to_run:
        for d, mode in enumerate(selected_modes):
            description_corpus = load_descriptions(mode, mult_lbl_enc.classes_)
            stopword_setting = 'ce' if mode == 'ICPC only' else 'mce'
            for m, method in enumerate(methods):
                #f1[m, d*3], _, _, preds_dev[mode][method], models[mode][method] = \
                #    run_distant_supervision(method, mode + '_rerun', description_corpus, y_desc, stopword_setting, dev_data, y_hot_dev,
                #                            mult_lbl_enc.classes_)
                
                descriptions_file = './results3/distant_descriptions.csv'
                f1_df = pd.DataFrame(f1, index=methods, columns=csv_header)
                f1_df.to_csv(descriptions_file, sep=',')

                # run without the A class
                f1[m, d * 3 + 1], _, _, _, _ = \
                    run_distant_supervision(method, mode + '_withoutA', description_corpus, y_desc[:, 1:], stopword_setting, dev_data, y_hot_dev[:, 1:],
                                            mult_lbl_enc.classes_[1:])

                f1_df = pd.DataFrame(f1, index=methods, columns=csv_header)
                f1_df.to_csv(descriptions_file, sep=',')

    # EXPERIMENT 3 -- without patient's speech ---------------------------
    # run selected methods with ICPC codes only, without patients' speech
    if '3' in args.experiment_to_run:

        f1 = np.zeros((len(methods), ncols))

        key = 'transcript__conversation_gp'
        for d, mode in enumerate(selected_modes):
            description_corpus = load_descriptions(mode, mult_lbl_enc.classes_)
            stopword_setting = 'ce' if mode == 'ICPC only' else 'mce'
            for m, method in enumerate(methods):
                f1[m, d*3 + 2], _, _, _, _ = run_distant_supervision(method, mode + '_gponly', description_corpus, y_desc, 'ce',
                                                               dev_data, y_hot_dev, mult_lbl_enc.classes_)

                descriptions_file = './results3/distant_descriptions_gponly.csv'
                f1_df = pd.DataFrame(f1, index=methods, columns=csv_header)
                f1_df.to_csv(descriptions_file, sep=',')

    # EXPERIMENT 4 -- run all methods with chosen descriptions and stopword setting -----------------------
    if '4' in args.experiment_to_run:

        # run a larger list of methods with ICPC codes only, with both sets of speech -- the best setup overall
        key = 'transcript__conversation_both'
        csv_header = 'F1 (dev), prec (dev), rec (dev), F1 (test), prec (test), rec (test)'
        results = np.zeros((len(methods), 6))
        for m, method in enumerate(methods):
            mode = 'ICPC only' if 'BERT' not in method else 'CKS only'
            description_corpus = load_descriptions(mode, mult_lbl_enc.classes_)
            if method not in preds_dev[mode]:
                results[m, 0], results[m, 1], results[m, 2], preds_dev[mode][method], models[mode][method] = \
                    run_distant_supervision(method, mode, description_corpus, y_desc, 'ce', dev_data, y_hot_dev, mult_lbl_enc.classes_)
            else:  # don't rerun the model, just load the predictions and compute results
                results[m, 0], results[m, 1], results[m, 2] = evaluate_classifications(y_hot_dev, preds_dev[mode][method],
                                                                                       mult_lbl_enc.classes_,
                                                                                       show_report=False)

            # repeat on test set
            results[m, 3], results[m, 4], results[m, 5], preds_test[mode][method], _ = run_distant_supervision(
                method, mode, description_corpus, y_desc, 'ce', test_data, y_hot_test, mult_lbl_enc.classes_,
                existing_model=models[mode][method])

            test_file = './results3/distant_test.csv'
            results_df = pd.DataFrame(results, index=methods, columns=csv_header.split(','))
            results_df.to_csv(test_file, sep=',')
        # np.savetxt(test_file, results, delimiter=',', fmt='%.3f', header=csv_header)

    # # EXPERIMENT 5 -- Save the per-class results for the best method -----------------------------------
    # perclass_file = './results3/distant_per_class_mnb.csv'
    # selected_method = 'multiclass NB'
    #
    # dev_f1, dev_prec, dev_rec = evaluate_per_class(y_hot_dev, preds_dev[mode][selected_method])
    # test_f1, test_prec, test_rec = evaluate_per_class(y_hot_test, preds_test[mode][selected_method])
    #
    # table = pd.DataFrame({
    #     'F1 (dev)': np.around(dev_f1, 3),
    #     'Prec (dev)': np.around(dev_prec, 3),
    #     'Rec (dev)': np.around(dev_rec, 3),
    #     'F1 (test)': np.around(test_f1, 3),
    #     'Prec (test)': np.around(test_prec, 3),
    #     'Rec (test)': np.around(test_rec, 3)
    # }, index=mult_lbl_enc.classes_)
    #
    # table.to_csv(perclass_file, sep=',')
    #
    # perclass_file = './results3/distant_per_class_mlm.csv'
    # selected_method = 'BERT mlm'
    #
    # dev_f1, dev_prec, dev_rec = evaluate_per_class(y_hot_dev, preds_dev[mode][selected_method])
    # test_f1, test_prec, test_rec = evaluate_per_class(y_hot_test, preds_test[mode][selected_method])
    #
    # table = pd.DataFrame({
    #     'F1 (dev)': np.around(dev_f1, 3),
    #     'Prec (dev)': np.around(dev_prec, 3),
    #     'Rec (dev)': np.around(dev_rec, 3),
    #     'F1 (test)': np.around(test_f1, 3),
    #     'Prec (test)': np.around(test_prec, 3),
    #     'Rec (test)': np.around(test_rec, 3)
    # }, index=mult_lbl_enc.classes_)
    #
    # table.to_csv(perclass_file, sep=',')

    # Using a model that was trained already, make predictions on the training set (the descriptions) to check for
    # overfitting
    f1 = np.zeros((len(methods), len(selected_modes)))

    if '6' in args.experiment_to_run:
        for d, mode in enumerate(selected_modes):
            description_corpus = load_descriptions(mode, mult_lbl_enc.classes_)
            stopword_setting = 'ce' if mode == 'ICPC only' else 'mce'
            for m, method in enumerate(methods):
                f1[m, d], _, _, _, _ = \
                   run_distant_supervision(method, mode + '_rerun', description_corpus, y_desc, stopword_setting,
                                           {'transcript__conversation_both':description_corpus}, y_desc, mult_lbl_enc.classes_)
                                           # existing_model=models[mode][method])

                descriptions_file = './results3/distant_descriptions_train.csv'
                f1_df = pd.DataFrame(f1, index=methods, columns=selected_modes)
                f1_df.to_csv(descriptions_file, sep=',')

