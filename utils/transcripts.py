from ast import literal_eval
from typing import List

from .preprocessing.text import utils_preprocess_text


def preprocess_transcripts(transcript, **kwargs):
    if type(transcript) == str:
        transcript = literal_eval(transcript)
    return [
        [speaker, utils_preprocess_text(text, **kwargs)]
        for (speaker, text) in transcript
    ]


def apply_to_transcript(transcript: List[List[str]], fn_speaker=None, fn_utterance=None, merge=False):
    new_transcript = []
    if not merge:
        for speaker, utterance in transcript:
            current = [speaker, utterance]
            if fn_speaker is not None:
                current[0] = fn_speaker(speaker)
            if fn_utterance is not None:
                current[1] = fn_utterance(utterance)
            new_transcript.append(current)
        return new_transcript
    else:
        speaker = [x[0] for x in transcript]
        utterance = [x[1] for x in transcript]
        if fn_speaker is not None:
            speaker = fn_speaker(speaker)
        if fn_utterance is not None:
            utterance = fn_utterance(utterance)
        return speaker, utterance


if __name__ == '__main__':
    tt = '[(\'GP\', \'How are you sir?\'), (\'Patient\', "I\'m not too bad at the moment actually."), (\'GP\', ' \
         '\'Right, okay good.\'), (\'Patient\', "I just thought I\'d come and touch base with you. A few weeks ago ' \
         'I had these little flares didn\'t I? That just flares up, and my hands aren\'t brilliant this morning ' \
         'but generally, because I\'m taking the 5mg of the-"), (\'Patient\', \'Take care.\'), (\'GP\', ' \
         '\'Goodbye.\'), (\'Patient\', \'Bye mate.\')] '
    y = preprocess_transcripts(tt)
    print(y)


def read_transcript(transcript, show_patient=True, show_gp=True, show_info=False, return_format='concat'):
    new_s = []
    new_t = []
    valid_speakers = []
    if show_patient:
        valid_speakers.extend(['pat'])  # patient
    if show_gp:
        valid_speakers.extend(['gp', 'doc'])  # doctor
    if show_info:
        valid_speakers.extend(['info'])

    if type(transcript) == str:
        transcript = literal_eval(transcript)
        
    for s, utt in transcript:
        valid = any(w in s.lower() for w in valid_speakers)
        if valid:
            new_s.append(s)
            new_t.append(utt)
    if return_format == 'concat':
        return ' '.join(new_t)
    elif return_format == 'list':
        return new_t
    else:
        return list(zip(new_s, new_t))