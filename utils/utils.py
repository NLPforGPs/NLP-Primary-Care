import os
import numpy  as np
from sklearn.preprocessing import MultiLabelBinarizer
import torch
import json

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
