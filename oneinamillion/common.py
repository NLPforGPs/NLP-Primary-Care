import os
from typing import List

import docx
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
        doc_text = '\n'.join(p_list)
        return doc_text
    else:
        raise NotImplementedError(f"cannot read {filename.rsplit('.', 1)[1]} formatted document")


def filter_word_docs(files: List[str]) -> List[str]:
    return list(filter(lambda _file: not _file.startswith('~$') and _file.find('.doc') > 0, files))


# # turns an object into dictionary recursively
# # https://stackoverflow.com/questions/1036409/recursively-convert-python-object-graph-to-dictionary
# def get_json_from_obj(obj, pretty=True):
#     if pretty:
#         return json.dumps(obj, default=lambda o: getattr(o, '__dict__', str(o)), sort_keys=True, indent=4)
#     else:
#         return json.dumps(obj, default=lambda o: getattr(o, '__dict__', str(o)), separators=(',', ':'))

if __name__ == '__main__':
    text = open_doc_as_txt(r"Z:\Transcripts\transcripts\01-11\20141208_125909-011110_Transcript.docx")
    print("test")
