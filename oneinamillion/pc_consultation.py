import logging
import os.path
from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from tqdm import tqdm

from oneinamillion.resources import PCC_PREPARED_PATH
from .primary_care.patient_record import PatientRecordParser, PatientRecords
from .primary_care.record_code import RecordCodeParser
from .primary_care.transcript import TranscriptParser, Transcript

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


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
    _seed = 0

    def __init__(self, seed=None):
        self._get_doc_ids()

        self._pd_path = os.path.join(PCC_PREPARED_PATH, 'consultation_PE.csv')

        if seed:
            self._seed = seed

    def get_sequence(self, record_ids: List[str]) -> [PrimaryCareDataPair]:
        for record_id in record_ids:
            yield self.get_single(record_id)

    def get_single(self, record_id: str) -> PrimaryCareDataPair:
        codes = self.record_code_parser.get(record_id)
        pt_record = self.record_parser.get(record_id)
        transcript = self.transcript_parser.get(record_id)
        return PrimaryCareDataPair(codes, pt_record, transcript)

    def _get_doc_ids(self):
        pt_record_list = self.record_parser.doc_ids
        transcript_list = self.transcript_parser.doc_ids
        mutual_ids = set(pt_record_list).intersection(transcript_list)
        self.doc_ids = list(mutual_ids)

        only_record = set(pt_record_list) - set(transcript_list)
        only_transcript = set(transcript_list) - set(pt_record_list)

        logger.info(f"Total primary care data-pairs: {len(self.doc_ids)}")
        logger.warning(f"The current IDs only have record documents.\n{np.sort(np.array(list(only_record)))}")
        logger.warning(f"The current IDs only have transcript documents.\n{np.sort(np.array(list(only_transcript)))}")

    # return list of indexes for train and test sets
    def create_train_test_split(self, n_splits=1, test_size=0.2):
        # codes = [self.record_code_parser.get(doc) for doc in self.doc_ids]
        flattened_pairs = []

        for doc in self.doc_ids:
            _codes = self.record_code_parser.get(doc)
            if _codes is None or str(_codes[0]) == 'nan':
                continue
            elif type(_codes) == list:
                for c in _codes:
                    if c[0] != '-':
                        flattened_pairs.append((doc, c))
            else:
                flattened_pairs.append((doc, _codes))

        doc_ids, codes = zip(*flattened_pairs)
        n_samples = len(doc_ids)
        doc_ids = np.array(doc_ids).reshape((n_samples, -1))
        doc_ids_flatten = doc_ids.flatten()
        classified_codes = [c[0] for c in codes]
        classified_codes = np.array(classified_codes).reshape((n_samples,))
        split = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=self._seed)
        for train_index, test_index in split.split(doc_ids, classified_codes):
            yield doc_ids_flatten[train_index], doc_ids_flatten[test_index], classified_codes[train_index],classified_codes[test_index]

    def get_pd(self, from_raw=False):
        if os.path.exists(self._pd_path) and not from_raw:
            df = pd.read_csv(self._pd_path, index_col=0)
            df.reset_index(inplace=True)
        else:
            # this will force a cache refresh from the two parsers
            self.record_parser.clear_cache()
            self.transcript_parser.clear_cache()
            ids = self.doc_ids
            columns = ['record_id', 'icpc_codes', 'pt_records',
                       'transcript__start_date', 'transcript__duration', 'transcript__conversation']
            df = pd.DataFrame(columns=columns)
            for ii, record_id in enumerate(tqdm(ids)):
                record = self.get_single(record_id)
                data = {
                    'record_id': record_id,
                    'icpc_codes': record.codes,
                    'pt_records': [vars(i) for i in record.pt_record.records],
                    'transcript__start_date': str(record.transcript.start_datetime),
                    'transcript__duration': str(record.transcript.duration),
                    'transcript__conversation': record.transcript.conversations,
                }
                df = df.append(data, ignore_index=True)
            df.to_csv(self._pd_path)
        return df


if __name__ == '__main__':
    pc = PCConsultation()
    x = pc.get_pd(from_raw=True)

