import numpy as np
import pandas as pd

from run_transcript_supervision import run_transcript_supervision
from utils.metrics.metric import evaluate_classifications
from classifier_wrappers import run_binary_naive_bayes, run_multiclass_naive_bayes, run_binary_svm, \
    run_multiclass_svm, run_nearest_centroid, run_nearest_neighbors, run_bert_conventional, run_mlm_classifier, \
    run_nsp_classifier
from prepare_data import prepare_original_data
from utils.utils import stratified_multi_label_split
from utils.stopwords import get_medical_stopwords, get_custom_stopwords, get_english_stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold


if __name__ == '__main__':
    # Load the data
    # YOU MAY NEED TO SET THE LOCATION OF THE DATASET IN oneinamillion/resources.py.
    orig_dataset, mult_lbl_enc, y_hot = prepare_original_data()
    id2label = mult_lbl_enc.classes_

    # Load our standard dev/test split
    dev_data, test_data, y_hot_dev, y_hot_test = stratified_multi_label_split(orig_dataset, y_hot)

    # Select which part of the transcripts we will use
    key = 'transcript__conversation_both'
    # key = 'transcript__conversation_gp'
    # key = 'transcript__conversation_patient'

    # EXPERIMENT 2 -- run all methods with chosen descriptions and stopword setting -----------------------

    methods = [
        'binary NB',
        'multiclass NB',
        'binary SVM',
        'multiclass SVM',
        'nearest centroid',
        'BERT MLM',
        'BERT NSP',
        'BERT conventional',
    ]
    # run a larger list of methods with ICPC codes only, with both sets of speech -- the best setup overall
    csv_header = 'F1 (train), prec (train), rec (train)'
    results = np.zeros((len(methods), 3))
    best_stopword_setting = 'mce'

    for m, method in enumerate(methods):
        # test on dev set with cross validation
        results[m, 0], results[m, 1], results[m, 2], _, _, _ = run_transcript_supervision(
            method, best_stopword_setting, dev_data, y_hot_dev, mult_lbl_enc.classes_, dev_data, y_hot_dev, seed=3)
        print(f'Results for {method} on train set. F1 = {results[m, 0]}')

        test_file = './results/supervised_train.csv'
        test_results = pd.DataFrame(np.around(results, 3), columns=csv_header.split(', '), index=methods)
        test_results.to_csv(test_file, sep=',')
