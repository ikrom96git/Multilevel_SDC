import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve

# Define parameters
c = 2.0  # Given constant
epsilon = 0.001  # Given epsilon
s = 0  # Assuming s = 0 for simplicity

# Compute ae and be
ae = (1 + np.sqrt(1 - 2 * c * epsilon**2)) / (2 * epsilon)
be = (1 - np.sqrt(1 - 2 * c * epsilon**2)) / (2 * epsilon)

# Initial conditions
x0 = np.array([1, 1, 1])  # Initial position
v0 = np.array([1, 1, 1])  # Initial velocity

# Time range
t_values = np.linspace(0, 5, 10000)  # Time from 0 to 10

# Define system of equations from initial conditions
t0 = 0  # Initial time

# Position equations at t = 0
eq1 = [0,0,0,0, np.cos(np.sqrt(c) * (t0 - s)), np.sin(np.sqrt(c) * (t0 - s))]  # x1(0)
eq2 = [np.sin(ae * (t0 - s)), -np.cos(ae * (t0 - s)), np.sin(be * (t0 - s)), -np.cos(be * (t0 - s)), 0, 0]  # x2(0)
eq3 = [np.cos(ae * (t0 - s)), np.sin(ae * (t0 - s)), np.cos(be * (t0 - s)), np.sin(be * (t0 - s)), 0, 0]  # x3(0)

# Velocity equations at t = 0
eq4 = [0, 0, 0, 0, -np.sqrt(c) * np.sin(np.sqrt(c) * (t0 - s)), np.sqrt(c) * np.cos(np.sqrt(c) * (t0 - s))]  # v1(0)
eq5 = [ae * np.cos(ae * (t0 - s)), ae * np.sin(ae * (t0 - s)), be * np.cos(be * (t0 - s)), be * np.sin(be * (t0 - s)), 0, 0]  # v2(0)
eq6 = [-ae * np.sin(ae * (t0 - s)), ae * np.cos(ae * (t0 - s)), -be * np.sin(be * (t0 - s)), be * np.cos(be * (t0 - s)), 0, 0]  # v3(0)

# Construct coefficient matrix
A = np.array([eq1, eq2, eq3, eq4, eq5, eq6])

# Right-hand side (initial conditions)
B = np.array([1, 1, 1, 1, 1, 1])

# Solve the system for unknowns a1, a2, b1, b2, c1, c2
solution = solve(A, B)

# Extract solutions
a1, a2, b1, b2, c1, c2 = solution

print(f"Computed coefficients:\n"
      f"a1 = {a1:.4f}, a2 = {a2:.4f}, b1 = {b1:.4f}, b2 = {b2:.4f}, c1 = {c1:.4f}, c2 = {c2:.4f}")

# Define the exact solution for position x_e(t)
def x_e(t):
    factor = c1 * np.cos(np.sqrt(c) * (t - s)) + c2 * np.sin(np.sqrt(c) * (t - s))
    
    x1 = a1 * np.sin(ae * (t - s)) - a2 * np.cos(ae * (t - s)) + b1 * np.sin(be * (t - s)) - b2 * np.cos(be * (t - s))
    x2 = a1 * np.cos(ae * (t - s)) + a2 * np.sin(ae * (t - s)) + b1 * np.cos(be * (t - s)) + b2 * np.sin(be * (t - s))
    
    return np.array([factor, x1, x2])

# Compute velocity as the derivative of x_e(t)
def v_e(t):
    factor_derivative = -c1 * np.sqrt(c) * np.sin(np.sqrt(c) * (t - s)) + c2 * np.sqrt(c) * np.cos(np.sqrt(c) * (t - s))
    
    x1_derivative = a1 * ae * np.cos(ae * (t - s)) + a2 * ae * np.sin(ae * (t - s)) + b1 * be * np.cos(be * (t - s)) + b2 * be * np.sin(be * (t - s))
    x2_derivative = -a1 * ae * np.sin(ae * (t - s)) + a2 * ae * np.cos(ae * (t - s)) - b1 * be * np.sin(be * (t - s)) + b2 * be * np.cos(be * (t - s))
    
    return np.array([factor_derivative, x1_derivative, x2_derivative])

# Compute values for plotting
x_values = np.array([x_e(t) for t in t_values])
v_values = np.array([v_e(t) for t in t_values])

# Plot position
plt.figure(figsize=(10, 5))
plt.plot(t_values, x_values[:, 0], label='x1')
plt.plot(t_values, x_values[:, 1], label='x2')
plt.plot(t_values, x_values[:, 2], label='x3')
plt.xlabel('Time')
plt.ylabel('Position')
plt.legend()
plt.title('Exact Solution for Position')
plt.grid()
plt.show()

# Plot velocity
plt.figure(figsize=(10, 5))
plt.plot(t_values, v_values[:, 0], label='v1')
plt.plot(t_values, v_values[:, 1], label='v2')
plt.plot(t_values, v_values[:, 2], label='v3')
plt.xlabel('Time')
plt.ylabel('Velocity')
plt.legend()
plt.title('Velocity Computed as Derivative')
plt.grid()
plt.show()
