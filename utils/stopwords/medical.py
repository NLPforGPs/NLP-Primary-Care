import os
dirname = os.path.dirname(__file__)
stopwords_file = os.path.join(dirname, 'medical_stopwords.txt')


def get_medical_stopwords():
    with open(stopwords_file, 'r') as file:
        stopwords = [line.strip() for line in file.readlines()]

        if 'side-effect' in stopwords:
            # stopwords.append('effect')
            # stopwords.append('effects')
            # stopwords.append('counter')
            # stopwords.append('side')
            stopwords.remove('side-effect')
            stopwords.remove('side-effects')
            stopwords.remove('over-the-counter')

        return stopwords


if __name__ == '__main__':
    print(get_medical_stopwords())
