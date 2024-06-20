"""
All of the problem for the second order problems can be found in:
file:///home/cwn4523/Downloads/978-3-319-18311-4.pdf
https://link.springer.com/book/10.1007/978-3-319-18311-4
The Duffing Equation is on page 93
"""
import numpy as np
from core.Pars import _Pars

class DuffingEquation(object):
    def __init__(self, problem_params):
        self.__name__='Duffing'
        self.params=_Pars(problem_params)
    
    def build_f(self, X, V, T):
        return -(self.params.omega**2)*X+self.params.eps*(V**2)*X

    def get_rhs(self, u, t):
        u=np.array(u)
        x_dot=u[1]
        v_dot=-(self.params.omega**2)*u[0]+self.params.eps*(u[1]**2)*u[0]
        return np.asarray([x_dot, v_dot])
    


    def get_exact_solution(self, t):
        pass
    
    def get_ntime_exact_solution(self, time):
        pass

class DuffingEquation_zeros_order_problem(object):
    def __init__(self, problem_params):
        self.params=_Pars(problem_params)
    
    def build_f(self, X, V, T):
        return -self.params.omega**2*X
    
    def get_rhs(self, u, t):
        u=np.array(u)
        x_dot=u[1]
        v_dot=-self.params.omega**2*u[0]
        return np.asarray([x_dot, v_dot])
    
    def get_exact_solution(self, t):
        M0=self.params.u0[0]
        N0=self.params.u0[1]
        x0=(N0/self.params.omega)*np.sin(self.params.omega*t)
        v0=N0*self.params.omega*np.cos(self.params.omega*t)
        return np.asarray([x0, v0])
    
    def get_ntime_exact_solution(self, time):
        solution=np.zeros([2, len(time)])
        for tt in range(len(time)):
            solution[:, tt]=self.get_exact_solution(time[tt])
        return solution
    
class DuffingEquation_first_order_problem(object):
    def __init__(self, problem_params):
        self.params=_Pars(problem_params)
    
    def build_f(self, X, V, T):
        a=2/self.params.omega
        const=0.25*(a**3)*(self.params.omega**2)*(np.sin(3*self.params.omega*T)+np.sin(self.params.omega*T))
        return -(self.params.omega**2)*X+const
    
    def get_rhs(self, u, t):
        u=np.array(u)
        a=2/self.params.omega
        const=-0.25*(a**3)*(self.params.omega**2)*(np.sin(3*self.params.omega*t)+np.sin(self.params.omega*t))
        x_dot=u[1]
        v_dot=-self.params.omega**2*u[0]+const
        return np.asarray([x_dot, v_dot])
    
    def asyp_expansion(self, zeros_order_sol, first_order_sol, eps):
        return zeros_order_sol+eps*first_order_sol

    def get_exact_solution(self, t):
        a=2/self.params.omega
        M1=0
        N1=(7*a**3)/32
        z=self.params.omega*t
        x1=M1*np.cos(z)+N1*np.sin(z)-((a**3)/32)*(4*z*np.cos(z)+np.sin(3*z))
        v1=-M1*self.params.omega*np.sin(z)+N1*self.params.omega*np.cos(z)-((a**3)/32)*(4*self.params.omega*np.cos(z)-4*z*self.params.omega*np.sin(z)+3*self.params.omega*np.cos(z))
        return np.asarray([x1, v1])
    
    def get_ntime_exact_solution(self, time):
        solution=np.zeros([2, len(time)])
        for tt in range(len(time)):
            solution[:, tt]=self.get_exact_solution(time[tt])
        return solution
        

