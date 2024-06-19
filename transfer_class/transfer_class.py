from abc import ABCMeta, abstractmethod

class transfer_class(ABCMeta):

    @abstractmethod
    def restriction_operator(self):
        pass

    @abstractmethod
    def fas_correction_operator(self):
        pass

    # @abstractmethod
    # def interpolation_operator(self):
    #     pass

    