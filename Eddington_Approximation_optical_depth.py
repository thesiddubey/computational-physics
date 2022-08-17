import matplotlib.pyplot as plt
import numpy as np

####################
# Plot of T vs s(kms) using the Eddington Approximation 
# finding Temperature in relation to the Optical depth
# Assuming Homogenous Atmosphere

# Set some convenient names
Te = 10000 # K
rho = 1.0e-6 # kg/m^3
kappa = 3.0 # m^2/kg

# Compute the optical depth, T, and s
tau = np.linspace(0.0,10.0,10000)
T = ( 3.0/4.0*Te**4 * (tau+2.0/3.0) )**0.25;
s = tau/(rho*kappa) / 1e3 # km

# Generate plot
plt.plot(s,T,'-b')
plt.xlim((0,1e3))
plt.ylim((0.5e4,1.5e4))
plt.xlabel(r'$s$ (km)')
plt.ylabel(r'$T$ (K)')

# Put some convient lines on the plot
plt.axhline(Te,color='r') # Horizontal line 
plt.axvline((2./3.)/(rho*kappa)/1e3,color='r') # Vertical line

plt.show()

####################
# Plotting f2(s) as a function of s(kms) where f2 is the fraction
# of hydrogen atoms in the first exvited state as a fn of temp
# assuming the photosphere above was electrically neutral 

# Set some constants
c = 2.997e8        # speed of light (m/s)
mp = 1.67e-27      # proton mass (kg)
k_eV = 8.61738e-5  # Boltzmann constant in eV/K
k = 1.381e-23      # Boltzmann constant in J/K
h = 6.626e-34      # Planck's constant (kg m^2/s)
me = 9.11e-31      # electron mass (kg);

# Determine fraction of neutral hydrogen in n=2 state
N2oN1 = 2.0**2 * np.exp( 13.6 * (1.0/2.0**2-1)/(k_eV*T) ) # Uses T from part a!
N2oN = N2oN1/(1+N2oN1)

# Use Saha to get the fraction of hydrogen that is neutral (see Q3)
alpha = mp/rho * (2.0*np.pi*me*k*T/h**2)**1.5 * np.exp(-13.6/(k_eV*T))
NIIoNt = 0.5*alpha*( np.sqrt(1+4./alpha) - 1 )
NIoNt = 1.0 - NIIoNt # Fraction of neutral hydrogen


# Finally, combine the two answers above to get the fraction of hydrogen in n=2 state
f2 = NIoNt * N2oN

# Plot as a function of depth
plt.plot(s,1e5*f2,'-b')
plt.xlabel(r'$s$~(km)')
plt.ylabel(r'$10^5\times f_2(s)$')
plt.xlim((0,1e3))

plt.show()

####################
# Plot of tau as a function of depth for Balmer 
# and non Balmer photons 

kappa_Balmer = 3.5e5

# Get the Balmer line opacities
ds = (s[1:]-s[:-1])*1e3 # in m
dtau_Balmer = rho*kappa*ds + rho*f2[:-1]*kappa_Balmer*ds
tau_Balmer = np.cumsum(dtau_Balmer)

# Plot as a function of depth
plt.plot(s[:-1],tau[:-1],'-b',label='non-Balmer')
plt.plot(s[:-1],tau_Balmer,'-r',label='Balmer')
plt.xlabel(r'$s$~(km)')
plt.ylabel(r'$\tau$')
plt.xlim((0,400))
plt.ylim((0,1))

plt.legend(loc='lower right')

plt.show()







