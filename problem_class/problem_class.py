from abc import ABCMeta, abstractmethod

class problem_class(ABCMeta):


    @abstractmethod
    def build_f(self):
        ...

    def get_rhs(self):
        ...

    