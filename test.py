import numpy as np
import matplotlib.pyplot as plt


###############################################
# Here, a simple test problem
T   	= 1.0  
tspan 	= np.array([0, T])
lambd  	= -1.
N 		= 100
K		= 100
EPSILON	= 0.001
t0=0
kappa_hat = 0.8
###############################################




###############################################
kappa=kappa_hat
c=1/EPSILON
F0=1/EPSILON
def get_exact_solution(tspan, u, t0):
      omega=np.sqrt(c-0.25*kappa**2)
      const_term=(c-1)/kappa
      Ap=(F0/kappa)*(1/(1+const_term**2))
      Bp=const_term*Ap
      pos_force=lambda time: Ap*np.sin(time)+Bp*np.cos(time)
      vel_force=lambda time: Ap*np.cos(time)-Bp*np.sin(time)
      b=np.array([u[0]-pos_force(t0), u[1]-vel_force(t0)])
      exp=lambda time: np.exp(-0.5*kappa*time)
      exp_cos=lambda time: exp(time)*np.cos(omega*time)
      exp_sin=lambda time: exp(time)*np.sin(omega*time)
      exp_cossin=lambda time: exp(time)*(-0.5*kappa*np.cos(omega*time)-omega*np.sin(omega*time))
      exp_sincos=lambda time: exp(time)*(-0.5*kappa*np.sin(omega*time)+omega*np.cos(omega*time))
      A=np.array([[exp_cos(t0), exp_sin(t0)], [exp_cossin(t0), exp_sincos(t0)]])
      a0, b0=np.linalg.solve(A, b)
      position=exp(tspan)*(a0*np.cos(omega*tspan)+b0*np.sin(omega*tspan))+pos_force(tspan)
      velocity=a0*exp_cossin(tspan)+b0*exp_sincos(tspan)+vel_force(tspan)
      return [position, velocity]

def get_zeros_order_solution(tspan, u, t0):
      b=np.array([u[0]-1, u[1]])
      A=lambda time: np.array([[np.sin(time), np.cos(time)], [np.cos(time), np.sin(time)]])
      a0, b0=np.linalg.solve(A(t0), b)
      position=a0*np.sin(tspan)+b0*np.cos(tspan)+1
      velocity=a0*np.cos(tspan)-b0*np.sin(tspan)
      return [position, velocity]

def get_first_order_solution(tspan, u, t0):
      b=np.array([u[0]-1, u[1]])
      A=lambda time: np.array([[np.sin(time), np.cos(time)], [np.cos(time), -np.sin(time)]])
      a0, b0=np.linalg.solve(A(t0), b)
      position=a0*np.sin(tspan)+b0*np.cos(tspan)+1
      velocity=a0*np.cos(tspan)-b0*np.sin(tspan)
      pos_force=lambda time: 0.5*kappa_hat*(a0*np.sin(time)+b0*np.cos(time))
      vel_force=lambda time: pos_force(time)+0.5*time*kappa_hat*(a0*np.cos(time)-b0*np.sin(time))
      B=lambda time: np.array([u[2]+time*pos_force(time), u[3]+vel_force(time)])
      a1, b1=np.linalg.solve(A(t0), B(t0))
      position_first=a1*np.sin(tspan)+b1*np.cos(tspan)-tspan*pos_force(tspan)
      velocity_first=a1*np.cos(tspan)-b1*np.sin(tspan)-vel_force(tspan)
      return [position, velocity, position_first, velocity_first]

def get_asyp_expansion(zeros, first, eps):
      return zeros+np.sqrt(eps)*first

    




if __name__=='__main__':
      time=np.linspace(0, 6, 1000)
      u=[2 ,0, 0, 0]
      t0=0
      solution=get_exact_solution(time, u, t0)
      macrotime=time/np.sqrt(EPSILON)

      zeros_order_solution=get_zeros_order_solution(macrotime, u, t0)
      first_order_solution=get_first_order_solution(macrotime, u, t0)
      position=get_asyp_expansion(first_order_solution[0], first_order_solution[2], eps=EPSILON)
      plt.plot(time, solution[0])
      plt.plot(time, zeros_order_solution[0])
      plt.plot(time, position)
      plt.show()