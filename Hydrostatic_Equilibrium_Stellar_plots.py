import matplotlib.pyplot as plt
import numpy as np

# For ODE integration
from scipy.integrate import odeint

# Deriviatives functions for Bonnor-Ebert sphere
# drho/dr = - M rho / r^2
# dM/dr = 4 pi r^2 rho
# 
# y[0] = rho
# y[1] = M
#
def dydr_func(r,y) :

    dydr = np.zeros(2)
    dydr[0] = - y[1]*y[0]/r**2
    dydr[1] = 4.0*np.pi*r**2*y[0]

    return dydr

# Generate a solution to the rationalized equations
r_tilde = np.linspace(1e-3,500,100000) # Start a small non-zero value so drho/dr doesn't explode
y0 = np.array([ 1.0, 4.0*np.pi/3.0 * r_tilde[0]**3 * 1.0 ] ) # Set to limiting small-r values
y = odeint(dydr_func,y0,r_tilde,tfirst=True)
rho_tilde = y[:,0] 
M_tilde = y[:,1] 

##############
# Plots of density and mass woth respect to the radius of the 
# isothermal gas spheres

# Plot the rationalized quantities
plt.figure(figsize=(3,2.5))
plt.plot(r_tilde,rho_tilde,'-b')
plt.xlim((0,5))
plt.ylim((0,1.05))
plt.xlabel(r'$\tilde{r}$')
plt.ylabel(r'$\tilde{\rho}$')
plt.tight_layout()
plt.show()

# Plot the rationalized quantities
plt.figure(figsize=(3,2.5))
plt.plot(r_tilde,M_tilde,'-b')
plt.xlim((0,5))
plt.ylim((0,12))
plt.xlabel(r'$\tilde{r}$')
plt.ylabel(r'$\tilde{M}$')
plt.tight_layout()
plt.show()

##############
# Plot of pressure vs radius for T = 10K and 1 Solar mass

M=1.989e30 # Solar mass in kg
T=10 # Temperature in K
mu = 2.4 # Mean molecular weight of molecular gas

G=6.67e-11 # Newton's constant in N m^2 / kg^2
k=1.38e-23 # Boltzmann's constant in J/K
mp = 1.67e-27 # Proton mass in kg

# Radius in AU
r = (r_tilde/M_tilde) * G*M*mu*mp/(k*T) / 1.496e11

# Density
rho = rho_tilde*(M/M_tilde)*(k*T*M_tilde/(G*M*mu*mp))**3

# Pressure
P = rho * k*T / (mu*mp)

# Plot the rationalized quantities
plt.figure(figsize=(3,2.5))
plt.plot(r/1e4,P,'-b')
plt.xlim((0,10))
plt.ylim((0,2e-12))
plt.xlabel(r'$r~({\rm 10^4 AU})$')
plt.ylabel(r'$P~({\rm Pa})$')
plt.tight_layout()
plt.show()


##############
# Plot of Jeans mass as a function of radius assuming
# average gas sphere density rhoavg

R = np.linspace(1e-5,1e5,1024)
rhoavg = M/(4.0*np.pi/3.0*(R*1.496e11)**3)
MJ = 0.2*(T/10)**1.5 * (rhoavg/3e-15)**-0.5

plt.figure(figsize=(3,2.5))
plt.plot(R/1e4,MJ,'-b')
plt.xlim((0,10))
plt.ylim((0,30))
plt.xlabel(r'$r~({\rm 10^4 AU})$')
plt.ylabel(r'$M_J~(M_\odot)$')
plt.tight_layout()
plt.show()

plt.figure(figsize=(3,2.5))
plt.plot(R/1e4,MJ,'-b')
plt.xlim((0,2))
plt.ylim((0,3))
plt.xlabel(r'$r~({\rm 10^4 AU})$')
plt.ylabel(r'$M_J~(M_\odot)$')
plt.tight_layout()
plt.show()
