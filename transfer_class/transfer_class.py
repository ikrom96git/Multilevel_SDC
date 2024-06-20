from abc import ABC, abstractmethod


class transfer_class(ABC):

    @abstractmethod
    def restriction_operator(self):
        pass

    @abstractmethod
    def fas_correction_operator(self):
        pass

    # @abstractmethod
    # def interpolation_operator(self):
    #     pass
