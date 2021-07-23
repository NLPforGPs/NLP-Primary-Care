import pandas as pd

from parser.patient_record import PatientRecordParser, PatientRecords
from parser.transcript import TranscriptParser, Transcript

PCC_CODES_FILE = r'Z:\codes\OiAM_ICPC-2.xlsx'


class PCConsultation:
    id_code_dict = {}
    record_parser = PatientRecordParser()
    transcript_parser = TranscriptParser()

    def __init__(self):
        self._get_pc_codes()

    def get_id_code_dict(self):
        return self.id_code_dict

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
                    if current_id in self.id_code_dict:
                        if type(self.id_code_dict[current_id]) != list:
                            self.id_code_dict[current_id] = [self.id_code_dict[current_id]]
                        self.id_code_dict[current_id].append(code)
                    else:
                        self.id_code_dict[current_id] = code

    def get(self, record_id: str) -> (PatientRecords, Transcript):
        pt_record = self.record_parser.get(record_id)
        transcript = self.transcript_parser.get(record_id)
        codes = self.id_code_dict[record_id]
        return pt_record, transcript



if __name__ == '__main__':
    pc = PCConsultation()
    pc._get_pc_codes()
    print(pc.get_id_code_dict())
