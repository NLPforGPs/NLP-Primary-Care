from utils.preprocessing.data import extract_icpc_categories
from utils.transcripts import preprocess_transcripts, read_transcript
from oneinamillion.pc_consultation import PCConsultation
from oneinamillion.resources import PCC_BASE_DIR, ICPC2CKS, PCC_CKS_DIR
import os
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
from oneinamillion.clinical_codes.icpc import IcpcParser
from utils.preprocessing.text import utils_preprocess_text
import pandas as pd
import numpy as np
from utils.preprocessing.data import write_path, segment_without_overlapping
import nltk
nltk.download('punkt')
from nltk import tokenize
from utils.preprocessing.text import cleaner
from oneinamillion.clinical_codes.cks import CksParser
from utils.utils import load_json_file
import random
import logging



def prepare_original_data():
    # alternatively, you may override the variables in oneinamillion.resources.py
    os.environ['PCC_BASE_DIR'] = "Z:/"

    print(f"RDSF base directory located at {PCC_BASE_DIR}")

    parser = PCConsultation()  # the only class needed to obtain all PC consultation data-pairs
    orig_dataset = parser.get_pd()

    # orig_dataset.head()  # uncomment to inspect the original dataset
    orig_dataset = orig_dataset.drop(orig_dataset[orig_dataset['icpc_codes']=="[nan]"].index)
    orig_dataset['codes'] = orig_dataset['icpc_codes'].apply(extract_icpc_categories)
    orig_dataset['transcript__conversation_clean'] = orig_dataset['transcript__conversation'].apply(
        preprocess_transcripts)
    orig_dataset['transcript__conversation_both'] = orig_dataset['transcript__conversation_clean'].apply(
        lambda t: read_transcript(t, return_format='concat'))
    orig_dataset['transcript__conversation_gp'] = orig_dataset['transcript__conversation_clean'].apply(
        lambda t: read_transcript(t, show_gp=True, show_patient=False, return_format='concat'))
    orig_dataset['transcript__conversation_patient'] = orig_dataset['transcript__conversation_clean'].apply(
        lambda t: read_transcript(t, show_gp=False, show_patient=True, return_format='concat'))

    # print(orig_dataset.head())
    # print(orig_dataset.info())

    y = orig_dataset['codes']
    mult_lbl_enc = MultiLabelBinarizer()
    y_hot = mult_lbl_enc.fit_transform(y)
    print(f"{len(mult_lbl_enc.classes_)} classification categories: {mult_lbl_enc.classes_}")

    return orig_dataset, mult_lbl_enc, y_hot


def load_icpc_descriptions():
    icpc_parser = IcpcParser()
    icpc_df = icpc_parser.get_pd()

    icpc_df['cat'] = icpc_df['Code'].astype('string').apply(lambda x: x[0].upper())

    clean_col = lambda x: utils_preprocess_text(x) if not pd.isna(x) else x
    # building keyword collection from three columns of the ICPC-2 descriptions
    icpc_df['criteria_prepared'] = icpc_df['criteria'].apply(clean_col)
    icpc_df['inclusion_prepared'] = icpc_df['inclusion'].apply(clean_col)
    icpc_df['preferred_prepared'] = icpc_df['preferred'].apply(clean_col)
    icpc_df['keywords'] = icpc_df[['preferred_prepared', 'criteria_prepared', 'inclusion_prepared']].fillna('').agg(
        ' '.join, axis=1)

    icpc_description_corpus = icpc_df[['cat', 'keywords']].groupby('cat').agg(' '.join).iloc[1:-1]
    icpc_description_corpus.index.name = None
    # icpc_description_corpus

    # print(f"dataset categories: {np.array(mult_lbl_enc.classes_).astype('str')}")
    # print(f"icpc descriptions:  {np.array(icpc_description_corpus.index).astype('str')}")

    return icpc_description_corpus


def load_cks_descriptions(cks_icpc=True):
    '''
    cks_icpc: True for coarse-grained labels, False for fine-grained labels
    '''
    # Integrate with CKS descriptions


    # use from_raw to refresh cached cks descriptions, and headings_to_include to use different set of sub-sections to
    # include
    cks_parser = CksParser()
    if cks_icpc:
        return cks_parser.get_pd()
    else:
        return cks_parser.get_cks_topic()

# def generate_fine_grained_descriptions():

#     df_fined_desc = load_cks_descriptions(cks_icpc=False) 



def load_descriptions(selected_mode, class_name=None):
    icpc_description_corpus = load_icpc_descriptions()
    cks_description_corpus = load_cks_descriptions()
    if class_name is None:
        class_name = icpc_description_corpus.index.tolist()
    # print(f"Description: {selected_mode}")
    icpc_description_dic = {}
    for icpc_code in class_name:
        icpc_code = icpc_code.upper()
        if selected_mode == 'ICPC only':
            icpc_description_dic[icpc_code] = f"{icpc_description_corpus.loc[icpc_code]['keywords']}"
        elif selected_mode == 'CKS only':
            icpc_description_dic[icpc_code] = f"{cks_description_corpus.loc[icpc_code]['cks descriptions']}"
        else:
            icpc_description_dic[icpc_code] = f"{icpc_description_corpus.loc[icpc_code]['keywords']} {cks_description_corpus.loc[icpc_code]['cks descriptions']}"

    icpc_corpus_df = pd.DataFrame.from_dict(icpc_description_dic, orient='index', columns=['keyword'])
    icpc_corpus = icpc_corpus_df['keyword']
    return icpc_corpus

def generate_descriptions(tokenizer, chunk_size, test_size, selected_mode, save_path, class_name=None, fine_grained=False):
    '''
    split descriptions into smaller chunks.
    '''
    train_data, test_data = [], []
    if fine_grained:
        descriptions = load_cks_descriptions(cks_icpc=False)
    else:
        descriptions = load_descriptions(selected_mode, class_name)
    for ii, _ in enumerate(descriptions):
        description = cleaner(descriptions[ii])
        sents = tokenize.sent_tokenize(description)
        chunks = segment_without_overlapping(tokenizer, sents, chunk_size)
        labels = [descriptions.index[ii]]*len(chunks)
        if len(chunks) <=2:
            continue
        train_examples, test_examples = train_test_split(list(zip(chunks, labels)), test_size=test_size,random_state=20211125)
        train_data.extend(train_examples)
        test_data.extend(test_examples)

    write_path(os.path.join(save_path, 'train.json'), train_data)
    write_path(os.path.join(save_path, 'test.json'), test_data)
        # processed_data.extend(zip(chunks, labels))
        # texts.extend(chunks)
        # all_labels.extend(labels)

# def generate_fine_grained_description(tokenizer, save_path):
#     '''
#     This function is used to generate fine-grained descriptions using CKS topics.
#     '''
#     train_data, test_data = [], []
#     descriptions = load_cks_descriptions(cks_icpc=False)
#     for ii, _ in enumerate(descriptions):
#         description = cleaner(descriptions[ii])
#         sents = tokenize.sent_tokenize(description)
#         chunks = segment_without_overlapping(tokenizer, sents, chunk_size)
#         labels = [descriptions.index[ii]]*len(chunks)
#         train_examples, test_examples = train_test_split(list(zip(chunks, labels)), test_size=test_size,random_state=20211125)
#         train_data.extend(train_examples)
#         test_data.extend(test_examples)
#     write_path(os.path.join(save_path, 'train.json'), train_data)
#     write_path(os.path.join(save_path, 'test.json'), test_data)

def generate_binary_descriptions(tokenizer, chunk_size, test_size, selected_mode, multiclass_desc_path, save_path, class_name=None, fine_grained=False):
    '''
    Generate data for binary classification.
    '''
    if not os.path.exists(multiclass_desc_path):
        print("Multiclass descriptions donot exist. Creating...")
        generate_descriptions(tokenizer, chunk_size, test_size, selected_mode, multiclass_desc_path, class_name, fine_grained=fine_grained)


    train_multiclass = os.path.join(multiclass_desc_path, 'train.json')
    test_multiclass = os.path.join(multiclass_desc_path, 'test.json')

    classmap_file = os.path.join(PCC_CKS_DIR, ICPC2CKS)
    classmap = load_json_file(classmap_file)
    # 
    class2class = {}
    if not fine_grained:
        for icpc1 in classmap:
            if icpc1 not in class2class:
                class2class[icpc1] = []
            for icpc2 in classmap:
                if set(classmap[icpc1]).intersection(set(classmap[icpc2])):
                    class2class[icpc1].append(icpc2)

    # nested list: [element, label]
    raw_train_data = load_json_file(train_multiclass)
    raw_test_data = load_json_file(test_multiclass)

    print('generating training data...')
    train_data = generate_binary_per_class(raw_train_data, class2class)
    print('generating test data...')
    test_data = generate_binary_per_class(raw_test_data, class2class)

    write_path(os.path.join(save_path, 'train.json'), train_data)
    write_path(os.path.join(save_path, 'test.json'), test_data)

def generate_binary_per_class(raw_data, class2class):
    '''
    sample negative examples for each category
    class2class: dictionary of class sharing the same health topic, which would be avioded when sampling
    raw_data: list of description , [[desc, label], ...]
    '''
    processed_data = []
    # dictionary of class to description
    class2desc = {}
    for item in raw_data:
        if item[1] not in class2desc:
            class2desc[item[1]] = []
        class2desc[item[1]].append(item[0])

    # fake mapping when there is no overlapping descriptions
    if not class2class:
        class2class = {key:[] for key in class2desc}

    for key in class2desc:
        num_examples = len(class2desc[key])
        # add postive examples
        processed_data.extend([[desc, key, 1] for desc in class2desc[key]])
        i = 0
        # sample negative examples
        while i < num_examples:
            selected = random.choice(raw_data)
            element = selected[:]
            element.append(0)
            # this example should not in category sharing the same description, not exist in processed data, should not from its own category
            if element[1] in class2class[key] or element in processed_data or element[1] == key:
                continue
            element[1] = key
            processed_data.append(element)
            i += 1 
    return processed_data

def prepare_transcripts_eval(tokenizer, save_path, max_length=490):
    parser = PCConsultation()
    orig_dataset = parser.get_pd()
    orig_dataset = orig_dataset.drop(orig_dataset[orig_dataset['icpc_codes']=="[nan]"].index)

    orig_dataset['codes'] = orig_dataset['icpc_codes'].apply(extract_icpc_categories)
    orig_dataset['transcript__conversation_both'] = orig_dataset['transcript__conversation'].apply(
        lambda t: read_transcript(t, return_format='list'))
    orig_dataset['transcript__conversation_both'] = orig_dataset['transcript__conversation_both'].apply(
        lambda t: segment_without_overlapping(tokenizer, t, max_length))
    data = list(zip(orig_dataset['transcript__conversation_both'].tolist(), orig_dataset['codes'].tolist(), orig_dataset['record_id'].tolist()))

    write_path(os.path.join(save_path, 'transcript.json'), data)


if __name__ == '__main__':

    tokenizer = AutoTokenizer.from_pretrained('microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract')
    save_path = './data/transcripts'
    # if not os.path.exists(save_path):
    # os.makedirs(save_path)
    # generate_descriptions(tokenizer, 490, 0.2, 'CKS only', save_path)
    prepare_transcripts_eval(tokenizer)