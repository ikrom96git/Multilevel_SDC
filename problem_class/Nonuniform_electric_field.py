import numpy as np
from problem_class.Uniform_electric_field import Uniform_electric_field

from core.Pars import _Pars

class Nonuniform_electric_field():

    def __init__(self, problem_params):
        self.prob=_Pars(problem_params)
        sqrt=np.sqrt(1-2*self.prob.c*(self.prob.epsilon**2))
        self.a_eps=(1+sqrt)/(2*self.prob.epsilon)
        self.b_eps=(1-sqrt)/(2*self.prob.epsilon)
        get_R_mat=Uniform_electric_field(problem_params)
        self.R_matrix=get_R_mat.R_matrix
        self.R_bar_matrix=get_R_mat.R_bar_matrix
    def get_matrix(self,t, s=0):
       
        # Initialize the matrix and parameters
        A = np.zeros((6, 6))
        c = np.sqrt(self.prob.c) * (t - s)
        a = self.a_eps * (t - s)
        b = self.b_eps * (t - s)

        # Fill the first three rows
        A[0, 4] = np.cos(c)
        A[0, 5] = np.sin(c)
        A[1, 0] = np.sin(a)
        A[1, 1] = -np.cos(a)
        A[1, 2] = np.sin(b)
        A[1, 3] = -np.cos(b)
        A[2, 0] = np.cos(a)
        A[2, 1] = np.sin(a)
        A[2, 2] = np.cos(b)
        A[2, 3] = np.sin(b)

        # Compute derivatives
        dc_dt = np.sqrt(self.prob.c)  # Derivative of c with respect to t
        da_dt = self.a_eps           # Derivative of a with respect to t
        db_dt = self.b_eps           # Derivative of b with respect to t

        # Fill rows 3 to 5 (derivatives with respect to t)

        # Row 3: Derivative of Row 1 with respect to t
        A[3, 4] = -np.sin(c) * dc_dt  # Derivative of cos(c)
        A[3, 5] = np.cos(c) * dc_dt   # Derivative of sin(c)

        # Row 4: Derivative of Row 2 with respect to t
        A[4, 0] = np.cos(a) * da_dt   # Derivative of sin(a)
        A[4, 1] = np.sin(a) * da_dt   # Derivative of -cos(a)
        A[4, 2] = np.cos(b) * db_dt   # Derivative of sin(b)
        A[4, 3] = np.sin(b) * db_dt   # Derivative of -cos(b)

        # Row 5: Derivative of Row 3 with respect to t
        A[5, 0] = -np.sin(a) * da_dt  # Derivative of cos(a)
        A[5, 1] = np.cos(a) * da_dt   # Derivative of sin(a)
        A[5, 2] = -np.sin(b) * db_dt  # Derivative of cos(b)
        A[5, 3] = np.cos(b) * db_dt   # Derivative of sin(b)
        
        return A
    
    def exact_solution(self, x_0, v_0, t, s=0):
        A_0=self.get_matrix(0, s)
        A=self.get_matrix(t, s)    
        b=np.concatenate((x_0, v_0))
        const=np.linalg.solve(A_0, b)
        # print(const)
        # breakpoint()
        return A@const
    
    def E_field(self, x):
        return self.prob.c*np.array([-x[0],0.5*x[1],0.5*x[2]])

    def asymp_solution(self, x_0, v_0, t, s=0):
        c=np.sqrt(self.prob.c)*(t-s)
        y0=np.array([x_0[0]*np.cos(c)+(v_0[0]/np.sqrt(self.prob.c))*np.sin(c), x_0[1], x_0[2]])
        u0=np.array([-x_0[0]*np.sqrt(self.prob.c)*np.sin(c)+v_0[0]*np.cos(c), v_0[1], v_0[2]])
        c_half=0.5*c*(t-s)
        y1=np.array([0, x_0[2]*c_half, -x_0[1]*c_half])
        u1=np.array([0, -v_0[2]*c_half, v_0[1]*c_half])
        theta=(t-s)/self.prob.epsilon
        A=np.concatenate((y0, self.R_matrix(theta)@u0))
        B=np.concatenate((y1+self.R_bar_matrix(theta)@u0, self.R_matrix(theta)@u1+self.R_bar_matrix(theta)@self.E_field(y0)))
        return A+self.prob.epsilon*B
    
        

