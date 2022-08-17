from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt


##################
# Derivatives for the Lorentz system
#
#   dx/dt = 10*(y-x)
#   dy/dt = x*(28-z)-y
#   dz/dt = x*y-8*z/3
#
#  x[] = [x,y,z]
#
def dxdt(t, x):

    dxdt = np.zeros(3)
    dxdt[0] = 10.0 * (x[1] - x[0])
    dxdt[1] = x[0] * (28.0 - x[2]) - x[1]
    dxdt[2] = x[0] * x[1] - 8.0 * x[2] / 3.0

    return dxdt


#################
# Integration via Euler's method
#
t = np.linspace(0, 5, 100000)  # A list of t values
# A list of [x,y,z] values the same length as t values
x = np.zeros((len(t), 3))

# Set the initial position
x[0, 0] = 5.0
x[0, 1] = 10.0
x[0, 2] = 15.0

# Step over t values and integrate via
#   x_{k+1} = x_k + dx/dt(x_k) * dt
for k in range(0, len(t) - 1):
    x[k + 1] = x[k] + dxdt(t[k], x[k]) * (t[k + 1] - t[k])

# Plot it up
plt.figure(figsize=(6, 6))
plt.plot(x[:, 0], x[:, 2], '-b', alpha=0.5, label='Euler')
plt.xlabel('x')
plt.ylabel('z')
plt.tight_layout()
plt.legend()
plt.savefig('Euler.png', dpi=300)


#################
# Integration via Runge-Kutta 2nd order method
#
xEuler = np.copy(x)  # Make a copy of the Euler's Method solution

# A list of t values (1/10th of the number of steps!)
t = np.linspace(0, 5, 10000)
# A list of [x,y,z] values the same length as t values
x = np.zeros((len(t), 3))

# Set the initial position
x[0, 0] = 5.0
x[0, 1] = 10.0
x[0, 2] = 15.0

# Step over t values and integrate via two steps:
#  1. x_{k+1/2} = x_k + dx/dt(x_k) * dt/2
#  2. x_{k+1} = x_k + dx/dt(x_{k+1/2}) * dt
for k in range(0, len(t) - 1):
    xh = x[k] + dxdt(t[k], x[k]) * 0.5 * (t[k + 1] - t[k])
    x[k + 1] = x[k] + dxdt(t[k], xh) * (t[k + 1] - t[k])

# Plot it up
plt.figure(figsize=(6, 6))
plt.plot(xEuler[:, 0], xEuler[:, 2], '-b', alpha=0.5, label='Euler')
plt.plot(x[:, 0], x[:, 2], '-r', alpha=0.5, label='RK2')
plt.xlabel('x')
plt.ylabel('z')
plt.tight_layout()
plt.legend()
plt.savefig('RK2.png', dpi=300)


#################
# Integration via SciPy Runge-Kutta 4th-5th order method
#
xRK2 = np.copy(x)  # Make a copy of the 2nd order Runge-Kutta Method solution

from scipy.integrate import odeint

# A list of t values (1/100th of the number of steps)
t = np.linspace(0, 5, 1000)
x0 = np.array([5.0, 10.0, 15.0])
x = odeint(dxdt, x0, t, tfirst=True)

# Plot it up
plt.figure(figsize=(6, 6))
plt.plot(xEuler[:, 0], xEuler[:, 2], '-b', alpha=0.5, label='Euler')
plt.plot(xRK2[:, 0], xRK2[:, 2], '-r', alpha=0.5, label='RK2')
plt.plot(x[:, 0], x[:, 2], '-g', alpha=0.5, label='odeint')
plt.xlabel('x')
plt.ylabel('z')
plt.tight_layout()
plt.legend()
plt.savefig('odeint.png', dpi=300)

# Save a data file
np.savetxt('lorentz_t.txt', t)  # t only
np.savetxt('lorentz_x.txt', x)  # x,y,z only
np.savetxt('lorentz_tx.txt', np.array(
    [t, x[:, 0], x[:, 1], x[:, 2]]).T)  # t and x,y,z

plt.show()
