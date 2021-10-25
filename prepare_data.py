from utils.preprocessing.data import extract_icpc_categories
from utils.transcripts import preprocess_transcripts, read_transcript
from oneinamillion.pc_consultation import PCConsultation
from oneinamillion.resources import PCC_BASE_DIR
import os
from sklearn.preprocessing import MultiLabelBinarizer

from oneinamillion.clinical_codes.icpc import IcpcParser
from utils.preprocessing.text import utils_preprocess_text
import pandas as pd
import numpy as np

from oneinamillion.clinical_codes.cks import CksParser


def prepare_original_data():
    # alternatively, you may override the variables in oneinamillion.resources.py
    os.environ['PCC_BASE_DIR'] = "Z:/"

    print(f"RDSF base directory located at {PCC_BASE_DIR}")

    parser = PCConsultation()  # the only class needed to obtain all PC consultation data-pairs
    orig_dataset = parser.get_pd()

    # orig_dataset.head()  # uncomment to inspect the original dataset

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


def load_icpc_descriptions(mult_lbl_enc):
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


def load_cks_descriptions():
    # Integrate with CKS descriptions

    # use from_raw to refresh cached cks descriptions, and headings_to_include to use different set of sub-sections to
    # include
    cks_parser = CksParser()
    cks_description_corpus = cks_parser.get_pd()
    # cks_description_corpus
    return cks_description_corpus


def load_descriptions(selected_mode, mult_lbl_enc):
    icpc_description_corpus = load_icpc_descriptions(mult_lbl_enc)
    cks_description_corpus = load_cks_descriptions()

    # print(f"Description: {selected_mode}")
    icpc_description_dic = {}
    for icpc_code in mult_lbl_enc.classes_:
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
