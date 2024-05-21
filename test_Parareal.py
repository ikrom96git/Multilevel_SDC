# Ignace Bossuyt, Python implementation of Parareal

#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy
import numpy.random
from math import sqrt
import copy

import numpy as np
import numpy.random as rdm

def Parareal(F, G, tspan, u0, N, K, C_F=1,C_G=1):
    
    def ITERATOR(x,y,z):
    	return x+y-z

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
        all_U[0,n+1,:]  = G([all_t[n], all_t[n+1]], all_U[0,n,:], k, n)
        G_old[n+1,:] = all_U[0,n+1,:]

    # parareal iteration 
    k_counter = 0
    for k in range(0,K):
        k_counter += 1
        for n in range(0,N):
            F_new[n+1,:] = F([all_t[n], all_t[n+1]], all_U[k,n,:],k,n)
        
        all_U[k+1,1,:] = u0 
        for n in range(0,N):
            G_new[n+1,:] = G([all_t[n], all_t[n+1]], all_U[k+1,n,:],k,n)
            all_U[k+1, n+1, :] = ITERATOR(F_new[n+1,:], G_new[n+1,:], G_old[n+1,:])
        G_old[:] = G_new
    
    ELAPSED_TIME = k_counter*(C_F+C_G) + C_G
#    all_errors = all_U .- all_U[end,:,:]'
#    all_errors_norm = [norm(all_errors[k,:,:]) for k=1:size(all_errors,1)]

    return (all_U[-1,-1,:], all_U[-1,:,:], ELAPSED_TIME, all_U) #, all_errors, all_errors_norm)    

############################################
def asyp_expansion(zeros_order_sol, first_order_sol, eps):
        return zeros_order_sol + np.sqrt(eps) * first_order_sol

def get_dy0(t, kappa_hat=0.8):
    a_0 = 0
    b_0 = 1

    dy = 0.5 * kappa_hat * (a_0 * np.sin(t) + b_0 * np.cos(t))
    dv = dy + 0.5 * kappa_hat * t * (a_0 * np.cos(t) - b_0 * np.sin(t))
    return t * dy, dv

def get_exact_solution_first_order(t, t0):
    dy0, dy1 = get_dy0(t)
    a1, b1 = get_dy0(t0)
    b1 = b1
    dy = a1 * np.sin(t) + b1 * np.cos(t) - dy0
    dv = a1 * np.cos(t) - b1 * np.sin(t) - dy1
    return np.array([dy, dv])

def get_exact_solution_zeros_order(t):
        a_0 = 0
        b_0 = 1

        dy = a_0 * np.sin(t) + b_0 * np.cos(t) + 1
        dv = a_0 * np.cos(t) - b_0 * np.sin(t)
        return np.array([dy, dv])

############################################
def get_cos(omega, t):
    return np.cos(omega * t)

def get_sin(omega, t):
    return np.sin(omega * t)

def get_dcos(omega, t):
    return -omega * np.sin(omega * t)

def get_dsin(omega, t):
    return omega * np.cos(omega * t)

def get_exp(omega, t):
    return np.exp(omega * t)

def get_dexp(omega, t):
    return omega * get_exp(omega, t)

def get_constForce(t, mu, kappa, F0, omega_force):

    c_omega = kappa - omega_force**2
    if mu != 0:
        const_omega = c_omega / (mu * omega_force)
        Ap = (F0 / (mu * omega_force)) * (1 / (1 + const_omega**2))
        Bp = const_omega * Ap
    else:
        raise (ValueError("Check get_constForce"))

    return Ap, Bp, c_omega


def get_forceTerm(t,mu, kappa, F0, omega_force):

    Ap, Bp, c_omega = get_constForce(t,mu, kappa,  F0, omega_force)
    position_force = Ap * get_sin(omega_force, t) + Bp * get_cos(
        omega_force, t
    )
    if c_omega == 0:
        velocity_force = Ap * get_dsin(omega_force, t) + (
            0.5 * F0 / (omega_force)
        ) * get_sin(omega_force, t)
    else:
        velocity_force = Ap * get_dsin(omega_force, t) + Bp * get_dcos(
            omega_force, t
        )

    return np.array([position_force, velocity_force])

def const_withoutFriction(t, mu, kappa,  F0=None, omega_force=None):
    omega = np.sqrt(kappa)
    A = np.zeros((2, 2))
    A[0, 0], A[0, 1] = get_cos(omega, t), get_sin(omega, t)
    A[1, 0], A[1, 1] = get_dcos(omega, t),get_dsin(omega, t)
    if omega_force is None:
        omega_force = 1.0

    forceterm = get_forceTerm(t, mu, kappa, F0=F0, omega_force=omega_force)
    u0 = u0 - forceterm

    b = np.linalg.solve(A, u0)
    return A, b

def const_withFriction(t, mu, kappa, F0=None, omega_force=None):
    determinant = mu**2 * 0.25 - kappa
    A = np.zeros((2, 2))
    omega_exp = -0.5 * mu
    exp = get_exp(omega_exp, t)
    dexp = get_dexp(omega_exp, t)
    omega = np.sqrt(-determinant)
    A[0, 0], A[0, 1] = exp * get_cos(omega, t), exp * get_sin(
        omega, t
    )
    A[1, 0] = dexp * get_cos(omega, t) + exp * get_dcos(omega, t)
    A[1, 1] = dexp * get_sin(omega, t) + exp * get_dsin(omega, t)

    
    if omega_force is None:
        omega_force = 1.0


    forceterm = get_forceTerm(t,mu, kappa,  F0=F0, omega_force=omega_force)

    u0 = u0 - forceterm
    b = np.linalg.solve(A, u0)

    return A, b


def get_solutionWithoutForce(t,t0, mu, kappa,  F0=None, omega_force=None):
    if mu == 0:
        *_, b = const_withoutFriction(t0,mu, kappa,  F0, omega_force)
        A, *_ = const_withoutFriction(t, mu, kappa)
    elif mu > 0:

        *_, b = const_withFriction(t0,mu, kappa,  F0, omega_force)
        A, *_ = const_withFriction(t, mu, kappa)
    else:
        raise ValueError("Solution without Force is not working")
    return A @ b


def get_solutionWithForce(t,t0, mu, kappa,  F0, omega_force=None):
    if omega_force is None:
        omega_force = 1.0
    solutionWithoutForce = get_solutionWithoutForce(t,mu, kappa,  F0, omega_force)

    forceTerm = get_forceTerm(t,mu, kappa,  F0, omega_force)
    solutionWithForce = solutionWithoutForce + forceTerm
    return solutionWithForce


###############################################
# Here, a simple test problem

T   = 1.0  # interval
r   = 0.05
sig = 0.2
K   = 100.0

STEP = 1000
NN = 8
M = 16
M0 = 4

tspan = [0, T]
hc = np.diff(tspan)[0]/NN/M0
hf = hc/M	

l = 1

P = 10**2
X0 = K*np.ones(P)

def F(tspan, u, k, n):
	Xf = u
	for n in range(0,M):
	    Xf[:] = (1.0 + r*hf)*Xf 
	return Xf
	      	
def G(tspan, u, k, n):
	Xc = u
	for n in range(0,M0):
	    Xc[:] = (1.0 + r*hc)*Xc
	return Xc		 

sth, U_final, COST, all_U = Parareal(F,G,tspan,X0,NN,l)
Xf = all_U[-1,-1,:]
Xc = all_U[-2,-1,:]	

print(X0)
print(all_U)
print(all_U.shape)
	
