import os
import re
from dataclasses import dataclass
from typing import List

import pandas as pd

"""
The data source for ICPC-2 codes:
https://www.ehelse.no/kodeverk/icpc-2e--english-version
"""

PCC_ICPC_DIR = r"Z:\codes\ICPC"
PCC_ICPC_FILE = "icpc-2e-v7.0.csv"


@dataclass
class IcpcCode:
    code: str
    preferred: str
    shortTitle: str
    inclusion: str
    inclusion_codes: List[str]
    exclusion: str
    exclusion_codes: List[str]
    criteria: str
    consider: str
    consider_codes: List[str]
    note: str
    icd10: str


class IcpcParser:
    def __init__(self):
        self.df: pd.DataFrame
        self.codes_list: List[IcpcCode] = []
        self.parse_code()

    def parse_code(self):

        def get_codes_from_str(text):
            if not pd.isna(text):
                return re.findall(r"[A-Z]\d{2}|-\d{2}", text)
            else:
                return text

        filename = os.path.join(PCC_ICPC_DIR, PCC_ICPC_FILE)
        df = pd.read_csv(filename).astype('string')

        df['inclusion_codes'] = df['inclusion'].apply(get_codes_from_str)
        df['exclusion_codes'] = df['exclusion'].apply(get_codes_from_str)
        df['consider_codes'] = df['consider'].apply(get_codes_from_str)

        self.df = df
        df.to_csv(os.path.join(PCC_ICPC_DIR, "icpc-prepared.csv"))

    def load(self):
        self.df = pd.read_csv(os.path.join(PCC_ICPC_DIR, "icpc-prepared.csv"))

    def get_objs(self):
        if not self.codes_list:
            codes = []
            for ii, row in self.df.iterrows():
                dct = row.to_dict()
                dct['code'] = dct['Code']
                dct.pop('Code')
                codes.append(
                    IcpcCode(**dct)
                )
            self.codes_list = codes
        return self.codes_list


if __name__ == '__main__':
    icpc = IcpcParser()
    icpc.get_objs()
