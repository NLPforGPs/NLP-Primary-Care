import logging

import pandas as pd

from oneinamillion.resources import PCC_CODES_FILE


class RecordCodeParser:
    _id_code_dict = {}

    def __init__(self):
        self._get_pc_codes()

    def _get_pc_codes(self):
        df: pd.DataFrame
        df = pd.read_excel(PCC_CODES_FILE, sheet_name='formatted').astype('string')

        for ii, row in df.iterrows():
            code = row['code']
            for i in range(1, 22):
                current_id: str
                current_id = row.get(i)
                if not pd.isna(current_id) and current_id.strip():
                    current_id = str(int(float(current_id)))  # to sanitize if cell is read as number with decimals
                    if current_id in self._id_code_dict:
                        if type(self._id_code_dict[current_id]) != list:
                            self._id_code_dict[current_id] = [self._id_code_dict[current_id]]
                        self._id_code_dict[current_id].append(code)
                    else:
                        self._id_code_dict[current_id] = [code]

    # lookup the ICPC codes for the record
    def get(self, record_id):
        code = self._id_code_dict.get(record_id)
        if code is None:
            code = self._id_code_dict.get(str(int(record_id)))  # strip leading 0s
        if code is None:
            logging.warning(f"{record_id} does not have ICPC code")
        return code
