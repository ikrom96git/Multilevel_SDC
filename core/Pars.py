class _Pars(object):
    def __init__(self, pars):
        for k, v in pars.items():
            setattr(self, k, v)
