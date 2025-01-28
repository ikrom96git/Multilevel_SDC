import numpy as np
import matplotlib.pyplot as plt
from core.Pars import Pars
class Uniform_electric_field:
    def __init__(self, problem_params):
        self.params=Pars(problem_params)

    def P_matrix(self):
        P=np.zeros((3, 3))
        P[0, 0]=1
        return P
    
    def R_matrix(self, theta):
        R=np.zeros((3,3))
        R[0,0]=1
        R[1,1]=np.cos(theta)
        R[1,2]=np.sin(theta)
        R[2,1]=-np.sin(theta)
        R[2,2]=np.cos(theta)
        return R
    
    def R_bar_matrix(self, theta):
        R_bar=np.zeros((3,3))
        R_bar[1,1]=np.sin(theta)
        R_bar[1,2]=1-np.cos(theta)
        R_bar[2,1]=np.cos(theta)-1
        R_bar[2,2]=np.sin(theta)
        return R_bar
    
    def A_matrix(self, t, epsilon,s=0):
        I=np.eye(3)
        O=np.zeros((3,3))
        theta=(t-s)/epsilon
        A=np.block([[I, (t-s)*self.P+epsilon*self.R_bar_matrix(theta)],[O, self.R_matrix(theta)]])
        return A
    
    def B_matrix(self, t, epsilon,s=0):
        b1=epsilon*(t-s)*np.array([0, -np.cos(t/epsilon), np.sin(t/epsilon)])
        b2=epsilon**2*np.array([0, np.sin(t/epsilon)-np.sin(s/epsilon), np.cos(t/epsilon)-np.cos(s/epsilon)])
        b3=(t-s)*np.array([0, np.sin(t/epsilon), np.cos(t/epsilon)])
        return np.block([b1+ b2, b3])
    
    def C_matrix(self, t, epsilon, s=0):
        theta=(t-s)/epsilon
        c1=epsilon*(t-s)*np.array([0, -np.cos(theta), np.sin(theta)])
        c2=(t-s)*np.array([0, np.sin(theta), np.cos(theta)])
        return np.block([c1, c2])

    def exact_solution(self, x_0, v_0, t, epsilon, s=0):
        A=self.A_matrix(t, epsilon, s)
        B=self.B_matrix(t, epsilon, s)
        u_0=np.concatenate((x_0, v_0))
        return A@u_0+B
    
    def Asymptotic_solution(self, x_0, v_0, t, epsilon, s=0):
        A=self.A_matrix(t, epsilon, s)
        C=self.C_matrix(t, epsilon, s)
        u_0=np.concatenate((x_0, v_0))
        return A@u_0+C
    



    


