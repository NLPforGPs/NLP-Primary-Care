import os
import numpy  as np
from sklearn.preprocessing import MultiLabelBinarizer
import torch
import json
from sklearn.model_selection import train_test_split
import pandas as pd


def save_predictions_to_file(y_test, y_pred_mat, filename):
    names = []
    data_dict = {}
    for l in range(y_test.shape[1]):
        names.append('gold_for_class_%i' % l)
        data_dict['gold_for_class_%i' % l] = y_test[:, l]
    for l in range(y_test.shape[1]):
        names.append('predictions_for_class_%i' % l)
        data_dict['predictions_for_class_%i' % l] = y_pred_mat[:, l]
    pred_df = pd.DataFrame(data_dict, columns=names)
    pred_df.to_csv('./output_predictions/%s' % filename)


def save_checkpoint(dir, epoch, name='checkpoint', **kwargs):
    state = {
        'epoch': epoch,
    }
    state.update(kwargs)
    if not os.path.exists(dir):
        os.makedirs(dir)

    filepath = os.path.join(dir, '%s.pt' % name)
    torch.save(state, filepath)


def softmax(logits):
    normalized = np.exp(logits - np.max(logits, axis = -1, keepdims=True))
    return normalized / np.sum(normalized, axis=-1, keepdims=True)


def merge_predictions(record_cnt, predictions, probs=False):
    cum_sum = np.cumsum(record_cnt)
    assert cum_sum[-1] == len(predictions)
    final_predictions = np.zeros((len(record_cnt), predictions.shape[1]))
    prev = 0

    for i in range(len(cum_sum)):
        entry = cum_sum[i]
        if probs:
            final_predictions[i] = np.max(predictions[prev:entry], axis=0)
            # final_predictions[i] = np.mean(predictions[prev:entry], axis=0)
        else:
            final_predictions[i] = np.any(predictions[prev:entry], axis=0).astype(int)
        prev = entry
    return final_predictions


def prediction_cks2icpc(map_file, predictions, label2name):
    '''
    Convert fine-grained predictions(cks topics) to icpc categories
    '''
    cks2icpc_dic = cks2icpc(map_file)
    # print(cks2icpc_dic)
    mapped_predictions = []
    for pred in predictions:
        labels_per_item = []
        for label in pred:
            labels_per_item.extend([label2name[icpc] for icpc in cks2icpc_dic[label] if icpc != 'Z'])
        mapped_predictions.append(labels_per_item)
    return mapped_predictions


def cks2icpc(map_filename):
    '''
    convert cks topics to icpc labels    
    '''
    with open(map_filename,'r', encoding='utf8') as f:
        icpc2cks = json.load(f)
    cks2icpc_dic = {}
    for icpc in icpc2cks:
        for cks in icpc2cks[icpc]:
            if cks not in cks2icpc_dic:
                cks2icpc_dic[cks] = []
            cks2icpc_dic[cks].append(icpc)
    return cks2icpc_dic


def one_hot_encode(labels, label2name):
    """
    convert labels to one-hot encoding
    labels: true labels of each transcript
    predict_labels: predicted labels of each transcript
    label2name: mapping from label to name
    """
    labels = [[label2name[code] for code in item] for item in labels]
    mult_lbl_enc = MultiLabelBinarizer()
    y_hot = mult_lbl_enc.fit_transform(labels)
    

    return y_hot, mult_lbl_enc


def select_supportive_sentence(probabilities, chunks, class_names, threshold):
    cat_ids = np.argmax(probabilities, axis=-1)
    pred_cats = class_names[cat_ids]
    probs = np.max(probabilities, axis=-1)
    
    #{cat1:[(sent1, prob)], cat2:[]}
    supportive_sentences = {}
    # zip(chunks, probs)

    for i in range(len(chunks)):
        if pred_cats[i] not in supportive_sentences:
            supportive_sentences[pred_cats[i]] = []
        supportive_sentences[pred_cats[i]].append((chunks[i], probs[i]))
    sorted_supportive_sentences = {}
    for cat in supportive_sentences:
        sorted_supportive_sentences[cat] = sorted(supportive_sentences[cat], key=lambda x: x[1], reverse=True)

    return sorted_supportive_sentences
    # map(lambda x: sorted(x.items()[1], key=x.items()[1][1], reverse=True), supportive_sentences)


def save_to_file(support_sentences, write_path):
    with open(write_path, 'w', encoding='utf8') as f:
        f.write('category'+'\t'+'chunks'+'\t'+'probability'+'\n')
        for cat in support_sentences:
            for item in support_sentences[cat]:
                f.write(cat+'\t'+ item[0] + '\t' + str(item[1])+'\n')


def load_json_file(data_path):
    with open(data_path, 'r', encoding='utf8') as f:
        data = json.load(f)
    return data


def stratified_multi_label_split(orig_dataset, y_hot, seed=23490, test_size=0.2, drop_undersize_classes=False):
    class_sizes = np.sum(y_hot, 0)
    class_idxs = np.argsort(class_sizes)  # iterate over the classes from smallest to largest

    np.random.seed(seed)
    random_states = np.random.randint(0, 2 ** 32 - 1, size=len(class_sizes))

    y_hot_tmp = np.copy(y_hot)  # copy the labels. To handle multi-label cases, we will deal with
    # each example according to its rarest class. Once it has been dealt with, the other labels must be zeroed out
    # so it is included only once.
    dev_data = None

    for cidx, c in enumerate(class_idxs):  # iterate over classes
        c_examples = np.argwhere(y_hot_tmp[:, c] == 1).flatten()  # find the members of each class
        y_hot_tmp[c_examples, :] = 0  # zero out the examples so they can't be chosen again

        if len(c_examples) < 2:  # can't split, too few instances. Merge with another small class
            if drop_undersize_classes:
                continue
            y_hot_tmp[c_examples, class_idxs[cidx+1]] = 1
            continue
        Xc_dev, Xc_test, yc_dev, yc_test = train_test_split(
            orig_dataset.iloc[c_examples] if isinstance(orig_dataset, pd.DataFrame) else np.array([orig_dataset[i] for i in c_examples]),
            y_hot[c_examples],
            test_size=test_size,
            train_size=1-test_size,
            random_state=random_states[cidx]
        )  # split the examples from this dataset

        if dev_data is not None:
            dev_data = dev_data.append(Xc_dev) if isinstance(dev_data, pd.DataFrame) else np.append(dev_data, Xc_dev, axis=0)
            test_data = test_data.append(Xc_test) if isinstance(test_data, pd.DataFrame) else np.append(test_data, Xc_test, axis=0)

            y_hot_dev = np.append(y_hot_dev, yc_dev, axis=0)
            y_hot_test = np.append(y_hot_test, yc_test, axis=0)
        else:
            dev_data = Xc_dev
            test_data = Xc_test

            y_hot_dev = yc_dev
            y_hot_test = yc_test

    return dev_data, test_data, y_hot_dev, y_hot_test


def convert_durations_to_count(durations):
    """
    Convert a list of duration strings to a list of counts of total seconds.
    :return: list of counts of total seconds
    """
    total_secs = []
    for duration in durations:
        toks = duration.split(':')
        secs = int(toks[0])*60*60 + int(toks[1])*60 + int(toks[2])
        total_secs.append(secs)
    print(total_secs)
    hist = np.histogram(total_secs)

    return total_secs, hist