from dataclasses import dataclass
from typing import List

from .primary_care.patient_record import PatientRecordParser, PatientRecords
from .primary_care.record_code import RecordCodeParser
from .primary_care.transcript import TranscriptParser, Transcript


@dataclass
class PrimaryCareDataPair:
    codes: List[str]
    pt_record: PatientRecords
    transcript: Transcript


class PCConsultation:
    record_parser = PatientRecordParser()
    transcript_parser = TranscriptParser()
    record_code_parser = RecordCodeParser()

    def get(self, record_id: str) -> (PatientRecords, Transcript):
        codes = self.record_code_parser.get(record_id)
        pt_record = self.record_parser.get(record_id)
        transcript = self.transcript_parser.get(record_id)
        return PrimaryCareDataPair(codes, pt_record, transcript)

    def report_match(self):
        pt_record_list = self.record_parser.get_doc_ids()
        transcript_list = self.transcript_parser.get_doc_ids()
        mutual = set(pt_record_list).intersection(transcript_list)
        only_record = set(pt_record_list) - set(transcript_list)
        only_transcript = set(transcript_list) - set(pt_record_list)
        return mutual, only_record, only_transcript


if __name__ == '__main__':
    pc = PCConsultation()
    print("test")
