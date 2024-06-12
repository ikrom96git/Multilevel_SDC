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
        
        self.params=_Pars(problem_params)
    
    def build_f(self, X, V, T):
        return -(self.params.omega**2)*X-self.params.eps*self.params.b*X**3

    def get_rhs(self, u, t):
        u=np.array(u)
        x_dot=u[1]
        v_dot=-self.params.omega**2*u[0]-self.params.eps*self.params.b*u[0]**3
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
        x0=M0*np.cos(self.params.omega*t)+N0*np.sin(self.params.omega*t)
        v0=-M0*self.params.omega*np.sin(self.params.omega*t)+N0*self.params.omega*np.cos(self.params.omega*t)
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
        const=-0.25*self.params.a**3*self.params.b*(np.cos(3*self.params.omega*T)+3*np.cos(self.params.omega*T))
        return -(self.params.omega**2)*X+const
    
    def get_rhs(self, u, t):
        u=np.array(u)
        const=-0.25*self.params.a**3*self.params.b*(np.cos(3*self.params.omega*t)+3*np.cos(self.params.omega*t))
        x_dot=u[1]
        v_dot=-self.params.omega**2*u[0]+const
        return np.asarray([x_dot, v_dot])
    
    def asyp_expansion(self, zeros_order_sol, first_order_sol, eps):
        return zeros_order_sol+eps*first_order_sol

    def get_exact_solution(self, t):
        M1=-(self.params.a**3*self.params.b)/(32*self.params.omega**2)
        N1=0
        z=self.params.omega*t
        x1=M1*np.cos(z)+N1*np.sin(z)+M1*(12*z*np.sin(z)-np.cos(3*z))
        v1=-M1*self.params.omega*np.sin(z)+N1*self.params.omega*np.cos(z)+M1*self.params.omega*(12*np.sin(z)+12*z*np.cos(z)+3*np.sin(3*z))
        return np.asarray([x1, v1])
    
    def get_ntime_exact_solution(self, time):
        solution=np.zeros([2, len(time)])
        for tt in range(len(time)):
            solution[:, tt]=self.get_exact_solution(time[tt])
        return solution
        

