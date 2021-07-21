import datetime
import os
from dataclasses import dataclass
from typing import List, Tuple, Any
from common import open_doc_as_txt, filter_word_docs
from tqdm import tqdm
from datetime import datetime, timedelta
import re

PCC_TRANSCRIPT_RAW_DIR = r"Z:\Transcripts\transcripts"
PCC_TRANSCRIPT_DIR = r"Z:\prepared\transcripts"

INFO = 'info'


class Transcript:
    id: str = None
    date_time: datetime = None
    duration: timedelta
    # contain speaker and spoken text
    conversations: List[Tuple[str, Any]]


@dataclass
class Transcript:
    id: str
    start_datetime: datetime
    duration: timedelta
    conversation: List[Any]


class TranscriptParser:

    @staticmethod
    def _parse_single_line(speaker: str, _line: str) -> List[Any]:
        leftover_content = _line.strip()
        conv = []
        while len(leftover_content) > 0:
            bracket_start_idx = leftover_content.find('[')
            bracket_end_idx = leftover_content.find(']')
            bracket_start = bracket_start_idx != -1
            bracket_end = bracket_end_idx != -1
            # contain ] matching from non-terminating [ from previous line
            if (bracket_start and bracket_end and bracket_end_idx < bracket_start_idx) or (bracket_end and not bracket_start):
                conv.append([INFO, leftover_content[:bracket_end_idx]])
                leftover_content = leftover_content[bracket_end_idx + 1:]
            # contain [ ] in sequence
            elif bracket_start and bracket_end and bracket_start_idx < bracket_end_idx:
                if bracket_start_idx > 0:
                    conv.append([speaker, leftover_content[:bracket_start_idx]])
                conv.append([INFO, leftover_content[bracket_start_idx + 1: bracket_end_idx]])
                leftover_content = leftover_content[bracket_end_idx + 1:]
            # contain only [ in sequence without matching ]
            elif bracket_start and not bracket_end:
                if bracket_start_idx > 0:
                    conv.append([speaker, leftover_content[:bracket_start_idx]])
                conv.append([INFO, leftover_content[bracket_start_idx + 1:]])
                leftover_content = ''
            # if contain no special [] symbols
            elif not bracket_start and not bracket_end:
                if len(leftover_content) > 1:  # prevents any redundant punctuations
                    conv.append([speaker, leftover_content])
                leftover_content = ''
        return conv

    # Parse the raw doc input into a Transcript object
    def _parse_transcript_doc(self, raw):
        lines: List[str]
        lines = raw.replace("\r\n", "\n").replace("\n\n", "\n").split("\n")  # remove extra blank lines
        lines = list(filter(lambda _line: len(_line) > 0, lines))

        # parse headers which includes File, Date, Time, Duration, ID, ...
        # they are optional to have

        if lines[0].find("Date") < 0 and lines[0].find("File") < 0:
            raise RuntimeError("No Date")
        date = lines[0].replace("Date:", "").replace("File:", "").strip()
        date = re.sub("[^0-9]", "", date)
        if len(date) == 6:
            date = f'20{date[4:6]}{date[2:4]}{date[0:2]}'
        if len(date) != 8:
            date = None
        if lines[1].find("Time") < 0:
            raise RuntimeError("No Time")
        time = lines[1].replace("Time:", "").strip()

        # make it compatible with special formatting in some documents
        if len(time) != 6 and ':' in time:
            time = f"{int(time.replace(':', '')):06d}"  # mm:ss
        elif len(time) == 5:
            time = f"{time[:4]}0{time[4:5]}" #hhmms
        if len(time) != 6:
            time = None

        if time is not None and date is not None:
            start_datetime = datetime.strptime(date + time, "%Y%m%d%H%M%S")
        elif date is not None:
            start_datetime = datetime.strptime(date, "%Y%m%d")
        else:
            start_datetime = None

        if lines[2].find("Duration") < 0:
            raise RuntimeError("No Duration")
        duration = lines[2].replace("Duration:", "").strip().split(':')
        if len(duration) == 3:
            duration = timedelta(hours=int(duration[0]), minutes=int(duration[1]), seconds=int(duration[2]))
        elif len(duration) == 2:
            duration = timedelta(minutes=int(duration[0]), seconds=int(duration[1]))
        else:
            raise NotImplementedError

        if lines[3].find("ID") >= 0:
            id = lines[3].replace("ID:", "").strip()
        elif lines[3].find("Date") >= 0:  # to accommodate some document may have ID labelled as Date
            id = lines[3].replace("Date:", "").strip()
        else:
            raise ValueError("id not found")

        def get_speaker_content(_line: str):
            segments = _line.strip().split(':', 1)
            if len(segments) == 2 and any(char.isalpha() for char in segments[0]):
                # if first segment contains only alphabets, then it is a speaker
                return segments[0].strip(), segments[1].strip()
            else:
                return None, _line.strip()


        line: str
        speaker = None
        conversation = []
        for line in lines[4:]:
            # print(line)
            new_speaker, content = get_speaker_content(line)
            if new_speaker is not None:
                speaker = new_speaker
            line_content = self._parse_single_line(speaker, content)

            # merge non-contiguous dialogue that came from the same speaker / INFO
            if len(line_content) > 0:
                first_new_dialogue = line_content[0]
                if len(conversation) > 0:
                    prev_dialogue = conversation[-1]
                    if first_new_dialogue[0] == prev_dialogue[0]:  # if they are the same speaker
                        conversation.pop()
                        line_content.pop(0)
                        new_dialogue = [prev_dialogue[0], prev_dialogue[1] + ' ' + first_new_dialogue[1]]
                        conversation.append(new_dialogue)

            conversation.extend(line_content)

        return Transcript(id, start_datetime, duration, conversation)

    def _save_txt(self, transcript: Transcript):
        if not os.path.exists(PCC_TRANSCRIPT_DIR):
            os.makedirs(PCC_TRANSCRIPT_DIR)
        filename = f"{transcript.id}_transcript.txt"
        target = os.path.join(PCC_TRANSCRIPT_DIR, filename)
        with open(target, "w") as text_file:
            text_file.write(f"{transcript.id}\n")
            text_file.write(f"{transcript.start_datetime}\n")
            text_file.write(f"{transcript.duration}\n")
            for dialogue in transcript.conversation:
                text_file.write(f"{dialogue[0]}\n")
                text_file.write(f"{dialogue[1]}\n")

    def _read_prepared_txt(self, filename) -> Transcript:
        target = os.path.join(PCC_TRANSCRIPT_DIR, filename)

        def sanitize(text: str):
            return ''.join(text.rsplit('\n', 1))

        with open(target) as text_file:
            id = sanitize(text_file.readline())
            dt_txt = sanitize(text_file.readline())
            if ':' in dt_txt:
                start_datetime = datetime.strptime(dt_txt, "%Y-%m-%d %H:%M:%S")
            else:
                start_datetime = datetime.strptime(dt_txt, "%Y-%m-%d")
            duration = sanitize(text_file.readline())
            digits = [float(d) for d in duration.split(':')]
            if len(digits) != 3:
                raise ValueError(f"{duration} cannot be parsed.")
            duration = timedelta(hours=digits[0], minutes=digits[1], seconds=digits[2])
            conversation = []
            line_is_speaker = True
            speaker = None

            for line in text_file:
                line = sanitize(line)
                if line_is_speaker:
                    speaker = line
                else:
                    conversation.append((sanitize(speaker), line))
                line_is_speaker = not line_is_speaker
            return Transcript(id, start_datetime, duration, conversation)

    def get(self):
        if not PCC_TRANSCRIPT_DIR:
            raise FileNotFoundError
        if not os.path.exists(PCC_TRANSCRIPT_DIR):
            raise FileNotFoundError
        files = os.listdir(PCC_TRANSCRIPT_DIR)
        txt_docs = list(filter(lambda _file: _file.find('.txt') > 0, files))
        _transcripts = []
        for file in txt_docs:
            transcript = self._read_prepared_txt(file)
            _transcripts.append(transcript)
        return _transcripts

    def prepare_raw(self):
        if not os.path.exists(PCC_TRANSCRIPT_RAW_DIR):
            raise FileNotFoundError
        dirs = os.listdir(PCC_TRANSCRIPT_RAW_DIR)
        for dir in tqdm(dirs):
            sub_path = os.path.join(PCC_TRANSCRIPT_RAW_DIR, dir)
            # only process valid word document
            word_docs = filter_word_docs(os.listdir(sub_path))
            for word_doc in tqdm(word_docs):
                text = open_doc_as_txt(os.path.join(sub_path, word_doc))
                parsed = self._parse_transcript_doc(text)
                self._save_txt(parsed)


if __name__ == '__main__':
    transcript_parser = TranscriptParser()
    transcript_parser.prepare_raw()
    transcript_parser.get()
