import logging
from dataclasses import dataclass
from typing import List

import numpy as np

from .primary_care.patient_record import PatientRecordParser, PatientRecords
from .primary_care.record_code import RecordCodeParser
from .primary_care.transcript import TranscriptParser, Transcript

from sklearn.model_selection import StratifiedShuffleSplit

logging.basicConfig(level=logging.DEBUG)


@dataclass
class PrimaryCareDataPair:
    codes: List[str]
    pt_record: PatientRecords
    transcript: Transcript


class PCConsultation:
    record_parser = PatientRecordParser()
    transcript_parser = TranscriptParser()
    record_code_parser = RecordCodeParser()

    doc_ids: list = []

    def __init__(self):
        self._get_doc_ids()

    def get_from_ids(self, record_ids: List[str]) -> [PrimaryCareDataPair]:
        for record_id in record_ids:
            yield self.get(record_id)

    def get(self, record_id: str) -> PrimaryCareDataPair:
        codes = self.record_code_parser.get(record_id)
        pt_record = self.record_parser.get(record_id)
        transcript = self.transcript_parser.get(record_id)
        return PrimaryCareDataPair(codes, pt_record, transcript)

    def _get_doc_ids(self):
        pt_record_list = self.record_parser.get_doc_ids()
        transcript_list = self.transcript_parser.get_doc_ids()
        mutual_ids = set(pt_record_list).intersection(transcript_list)
        self.doc_ids = list(mutual_ids)

        only_record = set(pt_record_list) - set(transcript_list)
        only_transcript = set(transcript_list) - set(pt_record_list)

        logging.info(f"Total primary care data-pairs: {len(self.doc_ids)}")
        logging.warning(f"The current IDs only have record documents.\n{np.sort(np.array(list(only_record)))}")
        logging.warning(f"The current IDs only have transcript documents.\n{np.sort(np.array(list(only_transcript)))}")

    # return list of indexes for train and test sets
    def create_train_test_split(self, n_splits=1):
        codes = [self.record_code_parser.get(doc) for doc in self.doc_ids]
        ll = []
        for _id, _codes in zip(self.doc_ids, codes):
            if type(_codes) == list:
                for c in _codes:
                    ll.append((_id, c))
            else:
                ll.append((_id, _codes))

        doc_ids, codes = zip(*ll)
        category_codes = (lambda xs: [x[0] for x in xs])(codes)

        stratified_shuffle = StratifiedShuffleSplit(n_splits=n_splits, test_size=0.2, random_state=0)
        doc_ids = np.array(doc_ids).reshape(-1, 1)
        category_codes = np.array(category_codes).reshape(-1, 1)
        splits = stratified_shuffle.split(doc_ids, category_codes)

        for train_split, test_split in splits:
            def _iloc(xs):
                return [doc_ids[x][0] for x in xs]

            yield _iloc(train_split), _iloc(test_split)


if __name__ == '__main__':
    pc = PCConsultation()
    print("test")
