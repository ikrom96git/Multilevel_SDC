# Ignace Bossuyt, Python implementation of Parareal

#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy
import numpy.random
from math import sqrt
import copy
from problem_class.HarmonicOscillator import HarmonicOscillator
from default_params.harmonic_oscillator_default_params import (
    get_harmonic_oscillator_default_params,
)
import numpy as np
import numpy.random as rdm


def ITERATOR(x, y, z):
    return x + y - z


def Parareal(F, G, R, L, tspan, u0, N, K, C_F=1, C_G=1):
    all_t = np.zeros(N + 1)
    for n in range(0, N + 1):
        all_t[n] = n / N * np.diff(tspan)[0]
    all_U = np.zeros((K + 1, N + 1, len(u0)))
    for k in range(0, K + 1):
        all_U[k, 0, :] = u0

    G_new = np.zeros((N + 1, len(u0)))
    G_old = np.zeros((N + 1, len(u0)))
    F_new = np.zeros((N + 1, len(u0)))

    # initial guess
    for n in range(0, N):
        all_U[0, n + 1, :] = L(G([all_t[n], all_t[n + 1]], R(all_U[0, n, :]), k, n))
        G_old[n + 1, :] = all_U[0, n + 1, :]

    # parareal iteration
    k_counter = 0
    for k in range(0, K):
        k_counter += 1
        for n in range(0, N):
            F_new[n + 1, :] = F([all_t[n], all_t[n + 1]], all_U[k, n, :], k, n)

        all_U[k + 1, 1, :] = u0
        for n in range(0, N):
            G_new[n + 1, :] = L(
                G([all_t[n], all_t[n + 1]], R(all_U[k + 1, n, :]), k, n)
            )
            all_U[k + 1, n + 1, :] = ITERATOR(
                F_new[n + 1, :], G_new[n + 1, :], G_old[n + 1, :]
            )
        G_old[:] = G_new

    return (all_U[-1, -1, :], all_U[-1, :, :], all_U)


###############################################
# Here, a simple test problem
T = 0.5
tspan = np.array([0, T])
lambd = -1.0
N = 10
K = 100
EPSILON = 0.001
t0 = 0
kappa_hat = 0.8
###############################################


###############################################
kappa = kappa_hat
c = 1 / EPSILON
F0 = 1 / EPSILON


###############################################


def get_exact_solution(tspan, u, t0):
    omega = np.sqrt(c - 0.25 * kappa**2)
    const_term = (c - 1) / kappa
    Ap = (F0 / kappa) * (1 / (1 + const_term**2))
    Bp = const_term * Ap
    pos_force = lambda time: Ap * np.sin(time) + Bp * np.cos(time)
    vel_force = lambda time: Ap * np.cos(time) - Bp * np.sin(time)
    b = np.array([u[0] - pos_force(t0), u[1] - vel_force(t0)])
    exp = lambda time: np.exp(-0.5 * kappa * time)
    exp_cos = lambda time: exp(time) * np.cos(omega * time)
    exp_sin = lambda time: exp(time) * np.sin(omega * time)
    exp_cossin = lambda time: exp(time) * (
        -0.5 * kappa * np.cos(omega * time) - omega * np.sin(omega * time)
    )
    exp_sincos = lambda time: exp(time) * (
        -0.5 * kappa * np.sin(omega * time) + omega * np.cos(omega * time)
    )
    A = np.array([[exp_cos(t0), exp_sin(t0)], [exp_cossin(t0), exp_sincos(t0)]])
    a0, b0 = np.linalg.solve(A, b)
    position = exp(tspan) * (
        a0 * np.cos(omega * tspan) + b0 * np.sin(omega * tspan)
    ) + pos_force(tspan)
    velocity = a0 * exp_cossin(tspan) + b0 * exp_sincos(tspan) + vel_force(tspan)
    return [position, velocity]


def get_zeros_order_solution(tspan, u, t0):
    b = np.array([u[0] - 1, u[1]])
    A = lambda time: np.array(
        [[np.sin(time), np.cos(time)], [np.cos(time), np.sin(time)]]
    )
    a0, b0 = np.linalg.solve(A(t0), b)
    position = a0 * np.sin(tspan) + b0 * np.cos(tspan) + 1
    velocity = a0 * np.cos(tspan) - b0 * np.sin(tspan)
    return [position, velocity]


def get_first_order_solution(tspan, u, t0):
    b = np.array([u[0] - 1, u[1]])
    A = lambda time: np.array(
        [[np.sin(time), np.cos(time)], [np.cos(time), -np.sin(time)]]
    )
    a0, b0 = np.linalg.solve(A(t0), b)
    position = a0 * np.sin(tspan) + b0 * np.cos(tspan) + 1
    velocity = a0 * np.cos(tspan) - b0 * np.sin(tspan)
    pos_force = lambda time: 0.5 * kappa_hat * (a0 * np.sin(time) + b0 * np.cos(time))
    vel_force = lambda time: pos_force(time) + 0.5 * time * kappa_hat * (
        a0 * np.cos(time) - b0 * np.sin(time)
    )
    B = lambda time: np.array([u[2] + time * pos_force(time), u[3] + vel_force(time)])
    a1, b1 = np.linalg.solve(A(t0), B(t0))
    position_first = a1 * np.sin(tspan) + b1 * np.cos(tspan) - tspan * pos_force(tspan)
    velocity_first = a1 * np.cos(tspan) - b1 * np.sin(tspan) - vel_force(tspan)
    return [position, velocity, position_first, velocity_first]


def F(tspan, u, k, n):
    tspan = np.diff(tspan)[0]
    t0 = tspan
    omega = np.sqrt(c - 0.25 * kappa**2)
    const_term = (c - 1) / kappa
    Ap = (F0 / kappa) * (1 / (1 + const_term**2))
    Bp = const_term * Ap
    pos_force = lambda time: Ap * np.sin(time) + Bp * np.cos(time)
    vel_force = lambda time: Ap * np.cos(time) - Bp * np.sin(time)
    b = np.array([u[0] - pos_force(t0), u[1] - vel_force(t0)])
    exp = lambda time: np.exp(-0.5 * kappa * time)
    exp_cos = lambda time: exp(time) * np.cos(omega * time)
    exp_sin = lambda time: exp(time) * np.sin(omega * time)
    exp_cossin = lambda time: exp(time) * (
        -0.5 * kappa * np.cos(omega * time) - omega * np.sin(omega * time)
    )
    exp_sincos = lambda time: exp(time) * (
        -0.5 * kappa * np.sin(omega * time) + omega * np.cos(omega * time)
    )
    A = np.array([[exp_cos(t0), exp_sin(t0)], [exp_cossin(t0), exp_sincos(t0)]])
    a0, b0 = np.linalg.solve(A, b)
    position = exp(tspan) * (
        a0 * np.cos(omega * tspan) + b0 * np.sin(omega * tspan)
    ) + pos_force(tspan)
    velocity = a0 * exp_cossin(tspan) + b0 * exp_sincos(tspan) + vel_force(tspan)
    return [position, velocity]


ORDER = 0
RESTRICTION = "classical"


def get_params(ORDER, RESTRICTION):
    if ORDER == 0:
        X0 = [2.0, 0.0]

        def G(tspan, u, k, n):
            tspan = np.diff(tspan)[0] / np.sqrt(EPSILON)
            t0 = tspan
            b = np.array([u[0] - 1, u[1]])
            A = lambda time: np.array(
                [[np.sin(time), np.cos(time)], [np.cos(time), np.sin(time)]]
            )
            a0, b0 = np.linalg.solve(A(t0), b)
            position = a0 * np.sin(tspan) + b0 * np.cos(tspan) + 1
            velocity = a0 * np.cos(tspan) - b0 * np.sin(tspan)
            return [position, velocity]

        def R(u, u_star=0):
            return u

        def L(u):
            return u

    elif ORDER == 1:
        X0 = [2.0, 0.0]

        def G(tspan, u, k, n):
            tspan = np.diff(tspan)[0] / np.sqrt(EPSILON)
            t0 = tspan
            b = np.array([u[0] - 1, u[1]])
            A = lambda time: np.array(
                [[np.sin(time), np.cos(time)], [np.cos(time), -np.sin(time)]]
            )
            a0, b0 = np.linalg.solve(A(t0), b)
            position = a0 * np.sin(tspan) + b0 * np.cos(tspan) + 1
            velocity = a0 * np.cos(tspan) - b0 * np.sin(tspan)
            pos_force = (
                lambda time: 0.5 * kappa_hat * (a0 * np.sin(time) + b0 * np.cos(time))
            )
            vel_force = lambda time: pos_force(time) + 0.5 * time * kappa_hat * (
                a0 * np.cos(time) - b0 * np.sin(time)
            )
            B = lambda time: np.array(
                [u[2] + time * pos_force(time), u[3] + vel_force(time)]
            )
            a1, b1 = np.linalg.solve(A(t0), B(t0))
            position_first = (
                a1 * np.sin(tspan) + b1 * np.cos(tspan) - tspan * pos_force(tspan)
            )
            velocity_first = a1 * np.cos(tspan) - b1 * np.sin(tspan) - vel_force(tspan)
            return [position, velocity, position_first, velocity_first]

        if RESTRICTION == "classical":

            def R(u, u_star=0):
                return [u[0], u[1], 0, 0.0]

        elif RESTRICTION == "optimisation":

            def R(u, u_star=0):
                raise Exception("to be implemented")

        def L(u):
            return np.array([u[0], u[1]]) + np.array(
                [sqrt(EPSILON) * u[2], sqrt(EPSILON) * u[3]]
            )

    return G, R, L, X0


EPS = np.finfo(float).eps

problem_params, time, eps = get_harmonic_oscillator_default_params("Fast_time")
model = HarmonicOscillator(problem_params)
time = np.linspace(0, T, N + 1)
ExactSolution = model.get_solution_ntimeWithForce(time)
# zeroth order
G, R, L, X0 = get_params(0, "anything")
sth, U_final, all_U = Parareal(F, G, R, L, tspan, X0, N, K)

# compute errors
all_errors = all_U - U_final
all_errors_norm = np.zeros(K + 1)
for k in range(0, K + 1):
    # all_errors_norm[k] = np.linalg.norm(np.mean(all_errors[k,:,:], axis=1))
    all_errors_norm[k] = np.linalg.norm(all_errors[k, :, :])

# # first order
G, R, L, X0 = get_params(1, "classical")
sth1, U_final1, all_U1 = Parareal(F, G, R, L, tspan, X0, N, K)

# compute errors
all_errors1 = all_U1 - U_final1
all_errors_norm1 = np.zeros(K + 1)
for k in range(0, K + 1):
    #    	all_errors_norm[k] = np.linalg.norm(np.mean(all_errors[k,:,:], axis=1))
    all_errors_norm1[k] = np.linalg.norm(all_errors1[k, :, :])
pos, vel = get_exact_solution(time, [2, 0], 0)
# plot the solution
fig, ax = plt.subplots()
ax.plot(np.linspace(0, T, N + 1), all_U[-1, :, 0], label="zeroth order")
ax.plot(np.linspace(0, T, N + 1), all_U1[-1, :, 0], label="first order")
# compute fine solution (evaluate the exact solution at the time points)
# in the figure, compare it with the last iterate of the zeroth order
ax.plot(np.linspace(0, T, N + 1), ExactSolution[0, :], label="exact", marker="o")
ax.legend()
# plot the errors
all_k = range(0, K + 1)
fig2, ax2 = plt.subplots()
ax2.semilogy(all_k, all_errors_norm + EPS, label="zeroth order")
ax2.semilogy(all_k, all_errors_norm1 + EPS, label="first order")
ax2.set_ylabel("error")
ax2.set_xlabel("iteration number")
ax2.set_title("Convergence of Parareal")
ax2.legend()

plt.show()
