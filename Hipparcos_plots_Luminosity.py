# Using sample data from Hipparcos satellite

import matplotlib.pyplot as plt
import numpy as np

# Read data
data = np.loadtxt('hipparcos.txt')

# Define some useful names
p = data[:, 0] * 1.0e-3  # in arcseconds
mv = data[:, 1]
mb = data[:, 2]
mi = data[:, 3]

# Get (B-V) color
bmv = mb - mv

# Get distance in pc
D = 1.0 / p

# Get absolute V-band magnitude
Mv = mv - 5.0 * np.log10(D / 10.0)


def plot_Mv(num):
    # Plotting absolute magnitude (Mv) as a function of (B-V) colour
    if(num == 1):
        # Generate scatter plot
        plt.plot(bmv, Mv, '.r')

        # Set the ranges
        plt.xlim((0.4, 1.7))
        plt.ylim((12, -1))

        # Add axis labels
        plt.xlabel('($B-V$)')
        plt.ylabel(r'$M_V$')

        plt.show()


def plot_TvsL(num):
    # Plotting log Luminosity as a function of log temperature of stars
    if(num == 1):
        # Convert to color temperature
        T = 9000 / (0.93 + bmv)

        # Convert to luminosity in units of solar luminosity
        L = 10**(-0.4 * (Mv - 4.83))

        # Generate scatter plot
        plt.plot(T, L, '.r')

        # Add lines for black bodies
        Tsun = 5770  # K
        Rsun = 6.955e8  # m
        sigma = 5.67e-8
        Lsun = 3.828e26  # W
        T = np.logspace(3, 4, 100)
        L = 4 * np.pi * sigma * Rsun**2 * T**4 / Lsun  # R=Rsun
        plt.plot(T, L, '-b')

        L = 4 * np.pi * sigma * (0.2 * Rsun)**2 * T**4 / Lsun  # R=0.2R_sun
        plt.plot(T, L, '--b')

        L = 4 * np.pi * sigma * (5.0 * Rsun)**2 * T**4 / Lsun  # R=5R_sun
        plt.plot(T, L, '--b')

        # Make axes logscale
        plt.xscale('log')
        plt.yscale('log')

        # Set the ranges
        plt.xlim((3e3, 1e4))
        plt.ylim((1e-4, 3e2))

        # Add axis labels
        plt.xlabel(r'$T$~(K)')
        plt.ylabel(r'$L/L_\odot$')

        plt.show()


# Starter Code (1 == Run)
plot_Mv(1)
plot_TvsL(1)
