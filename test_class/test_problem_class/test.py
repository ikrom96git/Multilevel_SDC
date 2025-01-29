import numpy as np
import matplotlib.pyplot as plt

def A_bar(y):
    y1, y2, y3 = y
    denom = y1**2 + y2**2
    return (1/denom)*np.array([[y2**2, -y1*y2 , 0.0],
                     [-y1*y2 , y1**2 , 0.0],
                     [0.0, 0.0, 0.0]])

def beta_bar(y, u):
    y1, y2, y3 = y
    u1, u2, u3 = u
    denom = y1**2 + y2**2
    return (1/denom)*np.array([u2 * (u1 * y2 - u2 * y1),
                     u1 * (u1 * y2 - u2 * y1),
                     0])

def ode_system(state):
    y = state[:3]
    u = state[3:]
    dydt = A_bar(y) @ u
    dudt = beta_bar(y, u)
    return np.concatenate((dydt, dudt))

def runge_kutta_4(f, y0, t):
    n = len(t)
    m = len(y0)
    y = np.zeros((n, m))
    y[0] = y0
    
    for i in range(n - 1):
        dt = t[i + 1] - t[i]
        k1 = f(y[i])
        k2 = f(y[i] + 0.5 * dt * k1)
        k3 = f(y[i] + 0.5 * dt * k2)
        k4 = f(y[i] + dt * k3)
        y[i + 1] = y[i] + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
    
    return y

# Initial conditions
y0 = np.array([1.0, 1.0, 1.0])
u0 = np.array([1, 0.01, 0.0])
initial_state = np.concatenate((y0, u0))

# Time span
t_span = (0, 5)
t_eval = np.linspace(t_span[0], t_span[1], 10000)

# Solve using RK4
solution = runge_kutta_4(ode_system, initial_state, t_eval)

# Extract results
y_values = solution[:, :3]
u_values = solution[:, 3:]

# Plot results
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(t_eval, y_values)
plt.xlabel('Time')
plt.ylabel('y values')
plt.legend(['y1', 'y2', 'y3'])
plt.title('Solution for y')

plt.subplot(1, 2, 2)
plt.plot(t_eval, u_values[:, :2])  # Only plot u1 and u2
plt.xlabel('Time')
plt.ylabel('u values')
plt.legend(['u1', 'u2'])
plt.title('Solution for u')

plt.tight_layout()
plt.show()

ax=plt.figure().add_subplot(projection='3d')
ax.plot(y_values[0], y_values[1], y_values[2], label='curve')
ax.legend()
plt.tight_layout()
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from mpl_toolkits.mplot3d import Axes3D

# Define the stiff system of ODEs
def stiff_system(t, state, epsilon):
    x, y, z = state
    dxdt = y
    dydt = -x + (z - 1) / epsilon
    dzdt = -y
    return [dxdt, dydt, dzdt]

# Parameters
epsilon = 0.1
T = 5
initial_state = [1, 0, 1]

# Time span
t_eval = np.linspace(0, T, 1000)

# Solve the ODE
sol = solve_ivp(stiff_system, [0, T], initial_state, t_eval=t_eval, args=(epsilon,), method='RK45')

# Extract solution
x, y, z = sol.y

# Plot the solution in 3D
fig = plt.figure(figsize=(6, 4))
ax = fig.add_subplot(111, projection='3d')

ax.plot(x, y, z, color='purple')

# Labels and title
ax.set_xlabel(r'$X(t)$')
ax.set_ylabel(r'$Y(t)$')
ax.set_zlabel(r'$Z(t)$')
ax.set_title(r'$\epsilon=0.01, T=5$')

# Show the plot
plt.show()
