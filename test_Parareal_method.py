# Ignace Bossuyt, Python implementation of Parareal

#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy
import numpy.random
from math import sqrt
import copy
from problem_class.HarmonicOscillator import HarmonicOscillator
from default_params.harmonic_oscillator_default_params import get_harmonic_oscillator_default_params
import numpy as np
import numpy.random as rdm

def ITERATOR(x,y,z):
	return x+y-z

def Parareal(F, G, R, L, tspan, u0, N, K, C_F=1,C_G=1):
    all_t = np.zeros(N+1) 
    for n in range(0,N+1):
    	all_t[n] = n/N*np.diff(tspan)[0]
    all_U = np.zeros((K+1,N+1,len(u0)))
    for k in range(0,K+1):
        all_U[k,0,:] = u0

    G_new = np.zeros((N+1,len(u0)))
    G_old = np.zeros((N+1,len(u0)))
    F_new = np.zeros((N+1,len(u0)))

    # initial guess
    for n in range(0,N):
        all_U[0,n+1,:]  = L(G([all_t[n], all_t[n+1]], R(all_U[0,n,:]), k, n))
        G_old[n+1,:] = all_U[0,n+1,:]

    # parareal iteration 
    k_counter = 0
    for k in range(0,K):
        k_counter += 1
        for n in range(0,N):
            F_new[n+1,:] = F([all_t[n], all_t[n+1]], all_U[k,n,:],k,n)
        
        all_U[k+1,1,:] = u0 
        for n in range(0,N):
            G_new[n+1,:] = L(G([all_t[n], all_t[n+1]], R(all_U[k+1,n,:]),k,n))
            all_U[k+1, n+1, :] = ITERATOR(F_new[n+1,:], G_new[n+1,:], G_old[n+1,:])
        G_old[:] = G_new
    
    return (all_U[-1,-1,:], all_U[-1,:,:], all_U)  


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
mu=kappa_hat
kappa = 1 / EPSILON
F0 = 1 / EPSILON
omega_force = 1.0



###############################################

def get_exact_solution(tspan, u, t0):
      omega=np.sqrt(kappa-mu**2/4)
      Ap=(F0/kappa)*(1+((kappa-1)/mu)**2)**(-1)
      Bp=((kappa-1)/mu)*Ap
      exp_sin=np.exp(-mu/2*t0)*np.sin(omega*t0)
      exp_cos=np.exp(-mu/2*t0)*np.cos(omega*t0)
      
      source_pos=Ap*np.sin(t0)+Bp*np.cos(t0)
      source_vel=Ap*np.cos(t0)-Bp*np.sin(t0)
      source_pos1=Ap*np.sin(tspan)+Bp*np.cos(tspan)
      source_vel1=Ap*np.cos(tspan)-Bp*np.sin(tspan)
      
      b=np.array([u[0]-source_pos,u[1]-source_vel])
      A=np.array([[exp_sin, exp_cos], [-omega*exp_sin-mu/2*exp_cos, omega*exp_cos-mu/2*exp_sin]])
      a_0, b_0 = np.linalg.solve(A, b)
      return [a_0*np.sin(tspan)+b_0*np.cos(tspan)+source_pos1, a_0*np.cos(tspan)-b_0*np.sin(tspan)+source_vel1]

def get_zeros_order_solution(tspan, u):
      v_scale=tspan/EPSILON
      b=np.array([u[0]-1,u[1]])
      A=np.array([[np.sin(v_scale), np.cos(v_scale)], [np.cos(v_scale), -np.sin(v_scale)]])
      a_0, b_0 = np.linalg.solve(A, b)
      return [a_0*np.sin(v_scale)+b_0*np.cos(v_scale)+1, a_0*np.cos(v_scale)-b_0*np.sin(v_scale)]

def get_first_order_solution(tspan, u):
      v_scale=tspan/EPSILON
      b=np.array([u[0]-1,u[1],0,0])
      A=np.array([[np.sin(v_scale), np.cos(v_scale)], [np.cos(v_scale), -np.sin(v_scale)]])
      a_0, b_0 = np.linalg.solve(A, b)
      y0_pos=0.5*kappa*(a_0*np.sin(v_scale)+b_0*np.cos(v_scale))
      y0_vel=y0_pos+0.5*kappa*v_scale*(a_0*np.cos(v_scale)-b_0*np.sin(v_scale))
      b1=np.array([u[2]-v_scale*y0_pos,u[3]-y0_vel])
      A1=np.array([[np.sin(v_scale), np.cos(v_scale)], [np.cos(v_scale), -np.sin(v_scale)]])
      a1, b1 = np.linalg.solve(A1, b1)
      y1_pos=a1*np.sin(v_scale)+b1*np.cos(v_scale)-v_scale*y0_pos
      y1_vel=a1*np.cos(v_scale)-b1*np.sin(v_scale)-y0_vel
      return [a_0*np.sin(v_scale)+b_0*np.cos(v_scale)+1, a_0*np.cos(v_scale)-b_0*np.sin(v_scale), y1_pos, y1_vel]


    



def F(tspan, u, k, n):
    tspan=np.diff(tspan)[0]
    omega=np.sqrt(kappa-mu**2/4)
    Ap=(F0/kappa)*(1+((kappa-1)/mu)**2)**(-1)
    Bp=((kappa-1)/mu)*Ap
    exp_sin=np.exp(-mu/2*tspan)*np.sin(omega*tspan)
    exp_cos=np.exp(-mu/2*tspan)*np.cos(omega*tspan)
    source_pos=Ap*np.sin(tspan)+Bp*np.cos(tspan)
    source_vel=Ap*np.cos(tspan)-Bp*np.sin(tspan)
    b=np.array([u[0]-source_pos,u[1]-source_vel])
    A=np.array([[exp_sin, exp_cos], [-omega*exp_sin-mu/2*exp_cos, omega*exp_cos-mu/2*exp_sin]])
    a_0, b_0 = np.linalg.solve(A, b)
    return [a_0*np.sin(tspan)+b_0*np.cos(tspan)+source_pos, a_0*np.cos(tspan)-b_0*np.sin(tspan)+source_vel]

	 
ORDER = 0
RESTRICTION = "classical"

def get_params(ORDER, RESTRICTION):
    if ORDER == 0: 
        X0 = [2., 0.]

        def G(tspan, u, k, n):
            v_scale=np.diff(tspan)[0]
            b=np.array([u[0]-1,u[1]])
            A=np.array([[np.sin(v_scale), np.cos(v_scale)], [np.cos(v_scale), -np.sin(v_scale)]])
            a_0, b_0 = np.linalg.solve(A, b)
            return [a_0*np.sin(v_scale)+b_0*np.cos(v_scale)+1, a_0*np.cos(v_scale)-b_0*np.sin(v_scale)]


        def R(u,u_star=0):
            return u

        def L(u):
            return u	 
		
    elif ORDER == 1:
        X0 = [2., 0.]
        def G(tspan, u, k, n):
            tspan=np.diff(tspan)[0]
            a_0 = u[1]
            b_0 = u[0]-1

            dy0 = a_0 * np.sin(tspan) + b_0 * np.cos(tspan) + 1
            dv0 = a_0 * np.cos(tspan) - b_0 * np.sin(tspan)
            dyc = 0.5 * kappa_hat * (a_0 * np.sin(tspan) + b_0 * np.cos(tspan))
            dvc = dyc + 0.5 * kappa_hat*tspan * (a_0 * np.cos(tspan) - b_0 * np.sin(tspan))

            x=np.array([u[2]+dyc, u[3]+dvc])
            A=np.array([[np.sin(tspan), np.cos(tspan)], [np.cos(tspan), -np.sin(tspan)]])
            a1, b1 = np.linalg.solve(A, x)
            print('Something is wrong here!')
            dy = a1 * np.sin(tspan) + b1 * np.cos(tspan) - tspan*dyc
            dv = a1 * np.cos(tspan) - b1 * np.sin(tspan) - dvc
            return [dy0, dv0, dy, dv]	 

        if RESTRICTION == "classical":
            def R(u, u_star=0):
                return [u[0], u[1], 0, 0.]
        elif RESTRICTION == "optimisation":	
            def R(u, u_star=0):
                raise Exception('to be implemented')

        def L(u):
            return np.array([u[0], u[1]]) + np.array([sqrt(EPSILON)*u[2], sqrt(EPSILON)*u[3]])
            
    return G, R, L, X0        

EPS = np.finfo(float).eps

problem_params, time, eps = get_harmonic_oscillator_default_params("Fast_time")
model = HarmonicOscillator(problem_params)
time=np.linspace(0, T, N+1)
ExactSolution=model.get_solution_ntimeWithForce(time)
# zeroth order
G, R, L, X0 = get_params(0, "anything")
sth, U_final, all_U = Parareal(F,G,R,L,tspan,X0,N,K)

# compute errors 
all_errors = all_U - U_final
all_errors_norm = np.zeros(K+1)
for k in range(0,K+1):
#    	all_errors_norm[k] = np.linalg.norm(np.mean(all_errors[k,:,:], axis=1))
	all_errors_norm[k] = np.linalg.norm(all_errors[k,:,:])

# # first order
# G, R, L, X0 = get_params(1, "classical")
# sth1, U_final1, all_U1 = Parareal(F,G,R,L,tspan,X0,N,K)

# # compute errors 
# all_errors1 = all_U1 - U_final1
# all_errors_norm1 = np.zeros(K+1)
# for k in range(0,K+1):
# #    	all_errors_norm[k] = np.linalg.norm(np.mean(all_errors[k,:,:], axis=1))
# 	all_errors_norm1[k] = np.linalg.norm(all_errors1[k,:,:])
pos, vel=get_exact_solution(time, [2,0], 0)
# plot the solution 
fig, ax = plt.subplots()
# ax.plot(np.linspace(0, T, N+1), all_U[-1,:,0], label='zeroth order')
# compute fine solution (evaluate the exact solution at the time points)
# in the figure, compare it with the last iterate of the zeroth order
ax.plot(np.linspace(0, T, N+1), ExactSolution[0,:], label='exact', marker='o')
ax.plot(np.linspace(0, T, N+1), pos, label='test order', marker='o')
ax.legend()
# plot the errors
all_k = range(0,K+1)
fig2, ax2 = plt.subplots()
ax2.semilogy(all_k, all_errors_norm + EPS, label='zeroth order')
# ax2.semilogy(all_k, all_errors_norm1 + EPS, label='first order')
ax2.set_ylabel('error')
ax2.set_xlabel('iteration number')
ax2.set_title('Convergence of Parareal')
ax2.legend()

plt.show()
	
