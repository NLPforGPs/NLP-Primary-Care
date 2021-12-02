# coding: utf-8
'''
@filename:prompt_ensebmle.py
@Createtime: 2021/12/02 09:40:09
@author: Haishuo Fang
@description: This script is used to ensemble models with different prompts
'''

from nn_model import DescClassifier
from transformers import AutoModelForMaskedLM, AutoTokenizer, BertConfig, BertForSequenceClassification
import os
import numpy as np
import argparse
from utils import load_data, merge_predictions, evaluate_classifications
from torch.utils.data import DataLoader
from dataset import DescLMDataset
from sklearn.preprocessing import MultiLabelBinarizer
from oneinamillion.resources import PCC_BASE_DIR
from datasets import load_dataset
from utils.preprocessing.data import prompt_encoding


if __name__ == __main__:
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, default='./model')
    # parser.add_argument('--model_list', type=str, default=)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--predict_data_dir', type=str, default='')
    parser.add_argument('--plm', type=str, default='microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext')

    args = parser.parse_args()

    model_dir = os.path.join(PCC_BASE_DIR, args.model_dir)
    predict_data_dir = os.path.join(PCC_BASE_DIR, args.predict_data_dir)

    model_lists = os.listdir(model_dir)

    dataset = load_dataset('../oneinamillionwrapper/transcript_evaldataset.py', download_mode="force_redownload", data_dir= predict_data_dir)
    dataset = dataset['test']
    splited_nums = dataset['split_nums']


    label2name = {'A':'general', 'B':'blood', 'D':'digestive','F':'eye', 'H':'ear', 'K':'cardiovascular','L':'musculoskeletal',
    'N':'neurological','P':'psychological','R':'respiratory','S':'skin','T':'endocrine', 'U':'urological','W':'pregnancy','X':'female',
    'Y':'male'}

    device = (torch.device('cuda') if torch.cuda.is_available()
                else torch.device('cpu'))

    pre_model = AutoModelForMaskedLM.from_pretrained(args.plm)
    tokenizer = AutoTokenizer.from_pretrained(args.plm)
    pre_model.to(device)

    classifier = DescClassifier(pre_model, 0, 0)

    all_logits =np.zeros((len(dataset['transcript']),len(label2name)))
    class_name = np.array(list(label2name.values()))
    for model_name in model_lists:
        
        checkpoint = torch.load(os.path.join(
                args.model_dir, model_name+'best-val-acc-model.pt'))
        classifier.load_state_dict(checkpoint['state_dict'])
        prompt = checkpoint['prompt']

        encoded_dataset = dataset.map(lambda e: prompt_encoding(e['transcript'], tokenizer, prompt), batched=True, remove_columns=['transcript', 'codes', 'split_nums'])
        encoded_dataset.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'targets'])

        predict_loader = DataLoader(encoded_dataset,  batch_size=args.batch_size, shuffle=False)

        _, logits = classifier.predict(predict_loader, device, tokenizer, class_name=class_name)
        all_logits += logits

    mean_logits = all_logits/len(model_lists)
    predictions = np.argmax(mean_logits, axis=-1)
    labels = class_name[selected_ids]
    labels = [[label] for label in labels]

    mult_lbl_enc = MultiLabelBinarizer()
    y_labels = [[label2name[code] for code in codes] for codes in dataset['codes']]
    y_hot = mult_lbl_enc.fit_transform(y_labels)
    predictions = mult_lbl_enc.transform(labels)
    final_predictions = merge_predictions(splited_nums, predictions)
    print(evaluate_classifications(y_hot, final_predictions, list(mult_lbl_enc.classes_)), show_report=True))

    # voting methods
    # merged_labels = []
    # for i in range(len(all_labels[0])):
    #     all_labels_per_segment = {}
    #     for j in range(len(model_lists)):
    #         all_labels_per_segment[all_labels[j][i][0]] = all_labels_per_segment.get(all_labels[j][i][0], 0) +1
    #     filtered = [key for key, value in all_labels_per_segment.items() if value>=2]
    #     merged_labels.append(filtered)

    # merged_labels_onehot = mult_lbl_enc.transform(merged_labels)
    # vote_predictions = merge_predictions(predict_data['splited_nums'], merged_labels_onehot)
    # print(evaluate_classifications(y_hot, final_predictions, list(label2name.values()), show_report=True))




        