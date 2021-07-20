import os
from typing import List

import docx
from docx import Document
import textract


def open_doc_as_txt(filename: str):
    if not os.path.exists(filename):
        return FileNotFoundError
    # you must install Antiword
    if filename.endswith('.doc'):
        raw_text = textract.process(filename, extension='doc').decode('utf-8')
        return raw_text
    elif filename.endswith('.docx'):
        doc = docx.Document(filename)
        p_list = [p.text for p in doc.paragraphs]
        text = '\n'.join(p_list)
        return text
    else:
        raise NotImplementedError(f"cannot read {filename.rsplit('.', 1)[1]} formatted document")


def filter_word_docs(files: List[str]) -> List[str]:
    return list(filter(lambda _file: not _file.startswith('~$') and _file.find('.doc') > 0, files))


if __name__ == '__main__':
    text = open_doc_as_txt(r"Z:\Transcripts\transcripts\01-11\20141208_125909-011110_Transcript.docx")
    print("test")
