from abc import abstractmethod


class Parser:

    @abstractmethod
    def prepare_raw(self):
        pass

    def read_prepared(self):
        pass