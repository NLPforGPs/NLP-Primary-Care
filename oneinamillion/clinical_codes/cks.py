import logging
import re
from os import path

import pandas as pd

from oneinamillion.resources import PCC_CODES_DIR, PCC_CODES_LINK_FILE, PCC_CKS_DIR, PCC_CKS_FILE
from utils.preprocessing.text import utils_preprocess_text, utils_remove_questions, utils_remove_bracket_sources

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

default_headings_to_include = [
    'Prevalence',
    'Prognosis',
    'Risk factors',
    'Differential diagnosis',
    'Diagnosis'
]


class CksParser:
    _link_dic = {}
    _cks_description_dic = {}
    _icpc_descriptions = {}
    _df = pd.DataFrame()

    def __init__(self, headings_to_include=None):
        self._get_cks()
        self._get_link()
        self._build_icpc_keywords_from_cks()
        self.headings_to_include = default_headings_to_include  # CKS sub-section headings to include
        if headings_to_include:
            self.headings_to_include = headings_to_include

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
            return any([x in heading for heading in self.headings_to_include])

        filtered_ids = cks_file['sub-section_topic'].apply(include_heading)
        cks_file = cks_file.loc[filtered_ids]

        def clean_text(text: str):
            return utils_preprocess_text(
                utils_remove_questions(utils_remove_bracket_sources(text))
            )

        cks_file['sub-section_text_prepared'] = cks_file['sub-section_text'].apply(clean_text)
        cks_file = cks_file[['topic', 'sub-section_text_prepared']].groupby('topic').agg(' '.join)
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

    def get_pd(self):
        """
        :return: the dataframe containing icpc-2 codes and the respective CKS descriptions
        """
        return self._df


if __name__ == '__main__':
    parser = CksParser()
    test = parser.get_pd()
