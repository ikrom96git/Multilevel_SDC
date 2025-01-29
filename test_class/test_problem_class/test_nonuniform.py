import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Parameters
eps = 0.01
c = 2

# Right-hand side of the ODE system
def ode_system(t, y):
    x, v = y[:3], y[3:]
    
    v_perp = np.array([0, v[2], -v[1]])
    E_x = c * np.array([-x[0], x[1] / 2, x[2] / 2])
    
    dxdt = v
    dvdt = (1 / eps) * v_perp + E_x
    
    return np.concatenate([dxdt, dvdt])

# Initial conditions: [x1, x2, x3, v1, v2, v3]
y0 = [1, 1, 1, 1, 1, 1]

# Time span
t_span = (0, 15)  # From t=0 to t=1
t_eval = np.linspace(*t_span, 10000)

# Solve the system
sol = solve_ivp(ode_system, t_span, y0, t_eval=t_eval, method='RK45')

# Plot results
fig, axes = plt.subplots(2, 1, figsize=(10, 6))
labels = ["x1", "x2", "x3"]
for i in range(3):
    axes[0].plot(sol.t, sol.y[i], label=labels[i])
    axes[1].plot(sol.t, sol.y[i+3], label=f"v{i+1}")

axes[0].set_title("Position Components")
axes[1].set_title("Velocity Components")
axes[0].legend()
axes[1].legend()
plt.tight_layout()
plt.show()
