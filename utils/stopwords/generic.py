import os

from sklearn.feature_extraction._stop_words import ENGLISH_STOP_WORDS

dirname = os.path.dirname(__file__)
stopwords_file = os.path.join(dirname, 'custom_stopwords.txt')


def get_custom_stopwords():
    with open(stopwords_file, 'r') as file:
        return [line.strip() for line in file.readlines()]


def get_english_stopwords():
    return list(ENGLISH_STOP_WORDS)


if __name__ == '__main__':
    print(get_custom_stopwords())
