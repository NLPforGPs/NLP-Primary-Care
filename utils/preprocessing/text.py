import re
import string
from typing import List

import nltk


def utils_preprocess_text(text: str, stemming: bool = False, lemmatisation: bool = True,
                          stopwords_list: List[str] = None, remove_stopwords: bool = True,
                          remove_punctuation: bool = False):
    # clean (convert to lowercase and remove punctuations and characters and then strip)

    if remove_punctuation:
        # add whitespace to punctuations without trailing spaces,
        # so words won't combine when punctuations are removed
        text = re.sub(r'([/.,?!:])([a-zA-Z])', r'\1 \2', text)
        # remove punctuations
        text = text.translate(str.maketrans('', '', string.punctuation))

    # Tokenize (convert from string to list)
    text_list = text.split()  # remove Stopwords
    if stopwords_list is not None and len(stopwords_list)>0:
        text_list = [word for word in text_list if word not in stopwords_list]

    if remove_stopwords:
        lst_stopwords = nltk.corpus.stopwords.words("english")
        text_list = [word for word in text_list if word not in lst_stopwords]

    # Stemming (remove -ing, -ly, ...)
    if stemming:
        ps = nltk.stem.porter.PorterStemmer()
        text_list = [ps.stem(word) for word in text_list]

    # Lemmatisation (convert the word into root word)
    if lemmatisation:
        lem = nltk.stem.wordnet.WordNetLemmatizer()
        text_list = [lem.lemmatize(word) for word in text_list]

    # back to string from list
    text = " ".join(text_list)
    return text


nltk.download('wordnet')
