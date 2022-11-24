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
import numpy as np
import pandas as pd
import scipy
from scipy.special import logsumexp
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.naive_bayes import MultinomialNB

from utils.metrics.metric import evaluate_classifications, evaluate_per_class
from classifier_wrappers import run_binary_naive_bayes, run_multiclass_naive_bayes, run_binary_svm, \
    run_multiclass_svm, run_nearest_centroid, run_nearest_neighbors, run_bert_classifier, run_bert_conventional, \
    run_mlm_classifier, run_nsp_classifier, label2name
from prepare_data import prepare_original_data
from utils.utils import stratified_multi_label_split
from prepare_data import load_descriptions
from utils.stopwords import get_medical_stopwords, get_custom_stopwords, get_english_stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt

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
                            classes, model=None):

    if 'BERT' in method:
        run_name = f"{run_name_template.lower().replace(' ', '_')}"

        if method == 'BERT conventional':
            clf = run_bert_conventional
        elif method == 'BERT MLM':
            clf = run_mlm_classifier
        elif method == 'BERT NSP':
            clf = run_nsp_classifier

        y_pred_mat_dev, _, model = clf(description_corpus, y_desc, id2label, dev_data[key], run_name, run_name_template, model)
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
        model = None

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
    selected_modes = ['ICPC only', 'CKS only']

    # EXPERIMENT 2  -- Comparing key methods with ICPC versus CKS -------------------
    # from the experiment one, the optimal setting is: ce for ICPC-2 and mce for CKS

    methods_for_description_test = [
        # 'binary NB',
        'multiclass NB',
        # 'nearest centroid',
        # 'BERT MLM',
        # 'BERT conventional',  # jdo we need this in final results?
        # 'BERT NSP',
    ]

    csv_header = []
    for mode in selected_modes:
        csv_header += [f'{mode}', f'{mode} without A', f'{mode} GP speech only']
    ncols = len(csv_header)

    f1 = np.zeros((len(methods_for_description_test), ncols))
    prec = np.zeros((len(methods_for_description_test), ncols))
    rec = np.zeros((len(methods_for_description_test), ncols))

    # cache some predictions and models for later
    preds_dev = {}  # first key is description, second is method
    preds_test = {}
    models = {}
    for mode in selected_modes:
        preds_dev[mode] = {}
        preds_test[mode] = {}
        models[mode] = {}

    # EXPERIMENT 4 -- run all methods with chosen descriptions and stopword setting -----------------------

    methods_for_best_descriptions = [
        # 'binary NB',
        'multiclass NB',
        # 'nearest centroid',
        # 'BERT MLM',
        # 'BERT conventional',
        # 'BERT NSP',
    ]

    # run a larger list of methods with ICPC codes only, with both sets of speech -- the best setup overall
    key = 'transcript__conversation_both'
    # csv_header = 'F1 (dev), prec (dev), rec (dev), F1 (test), prec (test), rec (test)'
    # results = np.zeros((len(methods_for_best_descriptions), 6))
    # for m, method in enumerate(methods_for_best_descriptions):
    #     mode = 'ICPC only' if 'BERT' not in method else 'CKS only'
    #     description_corpus = load_descriptions(mode, mult_lbl_enc.classes_)
    #     if method not in preds_dev[mode]:
    #         results[m, 0], results[m, 1], results[m, 2], preds_dev[mode][method], models[mode][method] = \
    #             run_distant_supervision(method, mode, description_corpus, y_desc, 'ce', dev_data, y_hot_dev, mult_lbl_enc.classes_)
    #     else:  # don't rerun the model, just load the predictions and compute results
    #         results[m, 0], results[m, 1], results[m, 2] = evaluate_classifications(y_hot_dev, preds_dev[mode][method],
    #                                                                                mult_lbl_enc.classes_,
    #                                                                                show_report=False)
    #
    #     # repeat on test set
    #     results[m, 3], results[m, 4], results[m, 5], preds_test[mode][method], _ = run_distant_supervision(
    #         method, mode, description_corpus, y_desc, 'ce', test_data, y_hot_test, mult_lbl_enc.classes_,
    #         model=models[mode][method])
    #
    #     test_file = './results/distant_test.csv'
    #     results_df = pd.DataFrame(results, index=methods_for_best_descriptions, columns=csv_header.split(','))
    #     results_df.to_csv(test_file, sep=',')
    #     # np.savetxt(test_file, results, delimiter=',', fmt='%.3f', header=csv_header)

    description_corpus = load_descriptions('ICPC only', mult_lbl_enc.classes_)
    model = MultinomialNB(alpha=0.001, fit_prior=False)
    description_vec, text_vectorizer = get_description_vectors(description_corpus, medical_stopwords+english_stopwords)
    model.fit(description_vec, np.argmax(y_desc, 1))

    # # Plot confusion matrix
    # method = 'multiclass NB'
    # confmat = confusion_matrix(np.argmax(y_hot_dev, 1), np.argmax(preds_dev['ICPC only'][method], 1))
    # disp = ConfusionMatrixDisplay(confmat, mult_lbl_enc.classes_)
    # disp.plot()
    # disp.figure_.savefig(f'./results/confmat_{method}.pdf')

    from wordcloud import WordCloud

    # # Explain a category -- show the keywords from the descriptions as a word cloud
    # text_vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words=medical_stopwords+english_stopwords, max_features=5000)
    # X = text_vectorizer.fit_transform(orig_dataset[key])
    # print(f"transcript bag-of-words+bigrams matrix shape: {X.shape}")
    vec_vocab = text_vectorizer.vocabulary_ # dictionary that contain the BOW tokens
    print(f"vocabulary size: {len(vec_vocab)}")
    #
    features_arr = np.array(text_vectorizer.get_feature_names())
    #
    # def explain_bow_vector(vec, ax):
    #     test = pd.DataFrame(vec, columns=features_arr).T.to_dict()[0]
    #     word_cloud = WordCloud(background_color="white").generate_from_frequencies(test)
    #     ax.imshow(word_cloud, interpolation='bilinear')
    #     ax.axis("off")
    #
    # plt.figure(figsize=(6, 40), dpi=300)
    # for i, cat in enumerate(mult_lbl_enc.classes_):
    #     ax = plt.subplot(len(mult_lbl_enc.classes_), 1, i+1)
    #     # get the total counts for all documents in the category
    #     X_i = np.sum(X[y_hot[:, i], :], axis=0)
    #     explain_bow_vector(X_i, ax)
    #     ax.title.set_text(cat + ": " + label2name[cat])
    #
    # plt.savefig('./results/wordclouds_transcripts.pdf')

    # Show probs learned by NB -- show the keywords from the descriptions as a word cloud
    def explain_bow_vector(vec, ax):
        test = pd.DataFrame(vec, columns=features_arr).T.to_dict()[0]
        word_cloud = WordCloud(background_color="white").generate_from_frequencies(test)
        ax.imshow(word_cloud, interpolation='bilinear')
        ax.axis("off")

    plt.figure(figsize=(6, 40), dpi=300)
    for i, cat in enumerate(mult_lbl_enc.classes_):
        ax = plt.subplot(len(mult_lbl_enc.classes_), 1, i+1)
        # get the total counts for all documents in the category

        X_i = np.exp(model.feature_log_prob_[i, :] - logsumexp(model.feature_log_prob_, axis=0))[None, :]
        explain_bow_vector(X_i, ax)
        ax.title.set_text(cat + ": " + label2name[cat])

    plt.savefig('./results/wordclouds_nb_icpc.pdf')
