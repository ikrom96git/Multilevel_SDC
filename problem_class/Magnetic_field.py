import numpy as np
from core.Pars import _Pars
from scipy.integrate import solve_ivp
from copy import deepcopy
class Magnetic_field():
    def __init__(self, prob_prams):
        self.prob=_Pars(prob_prams)
    
    def C_matrix(self, theta, x):
        sqrt_x=x[0]**2+x[1]**2
        C=np.zeros((3, 3))
        C[0,0]=(x[0]**2*np.cos(theta)+x[1]**2)/sqrt_x
        C[0,1]=(x[0]*x[1]*(np.cos(theta)-1))/sqrt_x
        C[0, 2]=-(x[0]*np.sin(theta))/np.sqrt(sqrt_x)
        C[1,0]=deepcopy(C[0,1])
        C[1, 1]=(x[1]**2*np.cos(theta)+x[0]**2)/sqrt_x
        C[1,2]=-(x[1]*np.sin(theta))/np.sqrt(sqrt_x)
        C[2,0]=-deepcopy(C[0,2])
        C[2,1]=-deepcopy(C[1,2])
        C[2,2]=np.cos(theta)
        return C
    def right_hand_side(self, t, y, eps):
        sqrt_x=1/(eps*np.sqrt(y[0]**2+y[1]**2))
        return [y[3], y[4], y[5], -sqrt_x*y[5]*y[0]+y[4],sqrt_x*y[5]*y[0]- y[3] ,sqrt_x*(-y[4]*y[0]+y[3]*y[2])] 
    
    def solve_ode(self, rhs,y0, time, eps):
        sol=solve_ivp(rhs, [0, time[-1]], y0, t_eval=time, args=(eps, ))
        return sol.y

    def solve_orginal(self, time):
        y0=self.prob.u0
        eps=self.prob.epsilon
        solution=self.solve_ode(self.right_hand_side, y0, time, eps)
        return solution





