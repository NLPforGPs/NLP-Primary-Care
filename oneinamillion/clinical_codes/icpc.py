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
        self.df: pd.DataFrame = pd.DataFrame()
        self._codes_list: List[IcpcCode] = []
        self._lookup: dict[str, IcpcCode] = {}

        self._parse_code()

    def _parse_code(self):
        def get_codes_from_str(text):
            if not pd.isna(text):
                return re.findall(r"[A-Z]\d{2}|-\d{2}", text)
            else:
                return []

        filename = os.path.join(PCC_ICPC_DIR, PCC_ICPC_FILE)
        df = pd.read_csv(filename).astype('string')

        df['inclusion_codes'] = df['inclusion'].apply(get_codes_from_str)
        df['exclusion_codes'] = df['exclusion'].apply(get_codes_from_str)
        df['consider_codes'] = df['consider'].apply(get_codes_from_str)

        self.df = df
        # df.to_csv(os.path.join(PCC_ICPC_DIR, "icpc-prepared.csv"))
        self._create_objs()
        self._create_lookup()
        self._link_child()

    def _create_objs(self):
        self._codes_list = []
        for row in self.df.itertuples(name='Code'):
            # noinspection PyUnresolvedReferences,PyProtectedMember
            dct = row._asdict()
            dct['code'] = dct['Code']
            dct.pop('Code')
            dct.pop('Index')
            self._codes_list.append(IcpcCode(**dct))

    def _create_lookup(self):
        self._lookup = {}
        for item in self._codes_list:
            self._lookup[item.code] = item

    def _link_child(self):
        for item in self._codes_list:
            def str2obj(codes) -> List[IcpcCode]:
                _codes = []
                for c in codes:
                    cc = self._lookup.get(c)
                    if cc is not None:
                        _codes.append(cc)
                return _codes

            item.inclusion_codes = str2obj(item.inclusion_codes)
            item.exclusion_codes = str2obj(item.exclusion_codes)
            item.consider_codes = str2obj(item.consider_codes)

    def get_objs(self):
        return self._codes_list


if __name__ == '__main__':
    icpc = IcpcParser()
    # icpc.get_objs()
    print("test")
