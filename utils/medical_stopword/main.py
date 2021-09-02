stopwords_file = 'stopwords.txt'


class MedicalStopwords:
    _stopwords = []

    def __init__(self):
        self._get_stopwords()

    def _get_stopwords(self):
        with open(stopwords_file, 'r') as file:
            self._stopwords = [line.strip() for line in file.readlines()]

    @property
    def get(self):
        return self._stopwords


if __name__ == '__main__':
    sw = MedicalStopwords()
    print(sw.get)
