import os
dirname = os.path.dirname(__file__)
stopwords_file = os.path.join(dirname, 'medical_stopwords.txt')


def get_medical_stopwords():
    with open(stopwords_file, 'r') as file:
        return [line.strip() for line in file.readlines()]


if __name__ == '__main__':
    print(get_medical_stopwords())
