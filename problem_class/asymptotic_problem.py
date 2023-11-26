import numpy as np
from problem_class.reduced_HO import Reduced_HO


class Slow_time(Reduced_HO):
    def __init__(self, prob_params):
        super().__init__(prob_params)

    def slow_time(self, t, order: int = 0):
        if order == 0:
            return np.cos(t)
        elif order == 1:
            return np.cos(t) + self.prob_params.kappa * np.sin(t)
        elif order == 2:
            return (1 - self.prob_params.kappa**2) * np.cos(
                t
            ) + 2 * self.prob_params.kappa * np.sin(t)
        else:
            raise ValueError("The order must be 0, 1 or 2")

    def slow_time_solution(self, t, order: int = 0):
        y0 = self.slow_time(t, order=0)
        y1 = y0 + self.prob_params.eps * self.slow_time(t, order=1)
        y2 = y1 + (self.prob_params.eps**2) * self.slow_time(t, order=2)
        if order == 0:
            return y0
        elif order == 1:
            return y1
        elif order == 2:
            return y2
        else:
            raise ValueError("The order must be 0, 1 or 2")

    def none_eps_solution(self, t):
        return np.cos(t)


class Fast_time(Reduced_HO):
    def __init__(self, prob_params):
        super().__init__(prob_params)

    def build_f(self, x, v, t):
        f = -1 * x + 0 * v + t
        return f
    def force(self,t):
        return np.ones(len(t))

    def convert_time(self, t):
        return t / np.sqrt(self.prob_params.eps)

    def fast_time(self, t, order: int = 0):
        time = self.convert_time(t)
        if order == 0:
            y0 = 0 * np.sin(time) + np.cos(time) + 1
        else:
            raise ValueError("The order must be 0")
        return y0
