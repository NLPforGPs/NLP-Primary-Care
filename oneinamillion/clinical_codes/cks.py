import logging
import re
from os import path
import json

import pandas as pd

from oneinamillion.resources import PCC_CODES_DIR, PCC_CODES_LINK_FILE, PCC_CKS_DIR, PCC_CKS_FILE, ICPC_CKS_DESC, ICPC2CKS, CKS_DESC
from utils.preprocessing.text import utils_preprocess_text, utils_remove_questions, utils_remove_bracket_sources

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

default_headings_to_include = [
#    'Prevalence',
#    'Prognosis',
#    'Risk factors',
#    'Differential diagnosis',
    'Causes',
    'Definition',
    'Diagnosis',
    'Clinical features',
    'History',
    'Presentation',
    'Signs and symptoms',
    'When to suspect'
]


class CksParser:
    _link_dic = {}
    _cks_description_dic = {}
    _icpc_descriptions = {}
    _df = pd.DataFrame()
    _cache_file_1 = path.join(PCC_CKS_DIR, ICPC_CKS_DESC)
    _cache_file_2 = path.join(PCC_CKS_DIR, CKS_DESC)
    _icpc2cks = path.join(PCC_CKS_DIR, ICPC2CKS)

    def __init__(self, headings_to_include=None, from_raw=False):
        """
        :param headings_to_include: sub-section headings to include from CKS description file
        :param from_raw: refresh cached CKS descriptions
        """
        self.headings_to_include = default_headings_to_include  # CKS sub-section headings to include
        if headings_to_include:
            self.headings_to_include = headings_to_include

        if not path.exists(self._cache_file_1) or not path.exists(self._cache_file_2):
            from_raw = True

        if from_raw:
            self._get_cks()
            self._get_link()
            self._build_icpc_keywords_from_cks()
            self._get_cks_topic()
        else:
            self._df = pd.read_csv(self._cache_file_1, index_col=0)
            self._df_fine_grained = pd.read_csv(self._cache_file_2, index_col=0)
            with open(self._icpc2cks, 'r') as f:
                self._link_dic = json.load(f)

    def _get_cks(self):
        """
        to obtain CKS symptoms and descriptions
        :return:
        """
        cks_file = path.join(PCC_CKS_DIR, PCC_CKS_FILE)
        if not path.exists(cks_file):
            raise FileNotFoundError(f"CKS file not accessible in {cks_file}")
        cks_file = pd.read_csv(cks_file)

        headings_to_exclude = []
        cks_file.columns = [re.sub(r'\s', '_', n.lower()) for n in list(cks_file.keys())]

        def include_heading(x):
            return any([str(x) in heading for heading in self.headings_to_include])

        filtered_ids = cks_file['sub-section_topic'].apply(include_heading)
        cks_file = cks_file.loc[filtered_ids]

        def clean_text(text: str):
            return utils_preprocess_text(
                utils_remove_questions(utils_remove_bracket_sources(text))
            )

        cks_file['sub-section_text_prepared'] = cks_file['sub-section_text'].apply(clean_text)
        cks_file = cks_file[['topic', 'sub-section_text_prepared']].groupby('topic').agg(' '.join)
        # {topic:concated text} 
        self._cks_description_dic = cks_file.to_dict()['sub-section_text_prepared']

    def _get_link(self):
        """
        to obtain the linkage between ICPC-2 codes to CKS codes
        :return:
        """
        # may need to convert CKS names to a unique ID
        link_file = path.join(PCC_CODES_DIR, PCC_CODES_LINK_FILE)
        if not path.exists(link_file):
            raise FileNotFoundError(f"ICPC-2 to CKS link file not accessible in {link_file}")
        link_file = pd.read_excel(link_file, index_col='Health Topic')
        # remove unwanted sum column
        if '*' in link_file:
            link_file = link_file.drop('*', axis=1)
        # rename column headers as ICPC code
        new_cols = [c[0].upper() for c in list(link_file.columns)]
        link_file.columns = new_cols
        # clean data as 0,1s
        for col in new_cols:
            link_file.loc[:, col] = link_file.loc[:, col].fillna(0).astype('int')
        link_file = link_file.T
        # create icpc to cks mapping (1-many)
        link_dic = {}
        for index, row in link_file.iterrows():
            link_dic[index] = [k for (k, v) in row.to_dict().items() if v == 1]
        # {'A':[topic1, topic2,..]}
        self._link_dic = link_dic

    def _build_icpc_keywords_from_cks(self):
        """
        to build a dataframe that contains icpc codes and their relevant CKS descriptions
        :return:
        """
        icpc_keys = self._link_dic.keys()

        not_found = False  # to indicate if their are headings mismatch
        for current_icpc in icpc_keys:
            current_cks_descriptions = ''
            cks_codes = self._link_dic[current_icpc]
            for cks_code in cks_codes:
                if cks_code in self._cks_description_dic:
                    current_cks_descriptions += f"{self._cks_description_dic[cks_code]} "
                else:
                    not_found = True
                    logger.warning(f"{cks_code} not found in cks descriptions.")
            self._icpc_descriptions[current_icpc] = current_cks_descriptions
        if not_found:
            logger.info(
                f"Please ensure that the title headings for CKS symptoms matches in both the cks and the link document.")
        self._df = pd.DataFrame.from_dict(self._icpc_descriptions, orient='index', columns=['cks descriptions'])
        self._df.to_csv(self._cache_file_1)

    def _get_cks_topic(self):
        self._df_fine_grained = pd.DataFrame.from_dict(self._cks_description_dic, orient='index', columns=['topics'])
        self._df_fine_grained.to_csv(self._cache_file_2)
        with open(self._icpc2cks, 'w') as f:
            json.dump(self._link_dic, f)

    def get_cks_topic(self):
        topics = set([t for topic in list(self._link_dic.values()) for t in topic])
        index = set(self._df_fine_grained.index.tolist())
        removal = list(index - topics)
        print(f'those topics {removal} not fit with icpc')
        filtered = self._df_fine_grained.drop(removal)

        return filtered['topics']


    def get_pd(self):
        """
        :return: the dataframe containing icpc-2 codes and the respective CKS descriptions
        """
        return self._df


if __name__ == '__main__':
    parser = CksParser()
    test = parser.get_pd()
