# For plotting
import matplotlib.pyplot as plt

# For file i/o and math
import numpy as np

####################
# Plot of Blamda vs lambda and 
# coinciding maxima with root using Bisection Method

# Define some constants.
k = 1.381e-23
h = 6.626e-34
c = 2.997e8

# Set temperature
T = 5500.0

# Specify the derivative of the Planck function to be zeroed.
def func_lambda(x):
    return x * np.exp(x) / (np.exp(x) - 1) - 5

# Find root via bisection
# Start with large range and initialize the list of values
xlist = np.array([0.01, 50.0, 100.0])
flist = func_lambda(xlist)

# Iterate many times, can adjust up and down to desired accuracy
for iteration in range(50):  # Try 50 iterations

    # If the root is between trials 0 and 1, set those to the limits
    if (flist[0] * flist[1] <= 0):
        xlist[2] = xlist[1]
        flist[2] = flist[1]

    # Otherwise the root is between trials 1 and 2, set those to the limits
    else:
        xlist[0] = xlist[1]
        flist[0] = flist[1]

    # Get the new trial at the midpoint
    xlist[1] = 0.5 * (xlist[0] + xlist[2])
    flist[1] = func_lambda(xlist[1])

xmax = xlist[1]

# Print wavelength of the maximum
lammax_nm = h * c / (xmax * k * T) * 1e9
print("Value of x at max", xmax)
print("Trial function value at maximum", flist[1])
print("Wavelength at max", lammax_nm)

# Plot the Planck function
# Make a list of wavelengths in nm and m
lam_nm = np.linspace(100, 2000, 500)
lam = lam_nm * 1e-9

# Compute the Planck function
Blambda = (2 * h * c**2 / lam**5) / (np.exp(h * c / (k * T * lam)) - 1)

# Plot the Planck function against lambda in nm
plt.plot(lam_nm, Blambda, '-b')

# Compute the wavlength at the max from the first part
lammax = lammax_nm * 1e-9
Blammax = (2 * h * c**2 / lammax**5) / (np.exp(h * c / (k * T * lammax)) - 1)

# Plot a a point on the
plt.plot(lammax_nm, Blammax, 'or')

# Add some labels
plt.xlabel(r'$\lambda$ (nm)')
plt.ylabel(r'$B_\lambda$ (W/m$^3$ Sr)')

# Add title
plt.title(r'$B_\lambda$ vs $\lambda$')

plt.show()

####################
# Plot of Bfreq vs freq and coinciding 
# with where the frequency peaks

# Define some constants.
k = 1.381e-23
h = 6.626e-34
c = 2.997e8

# Set temperature
T = 5500.0

# Specify the derivative of the Planck function to be zeroed.
def func_freq(x):
    return x * np.exp(x) / (np.exp(x) - 1) - 3

# Find root via bisection
# Start with large range and initialize the list of values
xlist = np.array([0.01, 50.0, 100.0])
flist = func_freq(xlist)

# Iterate many times, can adjust up and down to desired accuracy
for iteration in range(50):  

    # If the root is between trials 0 and 1, set those to the limits
    if (flist[0] * flist[1] <= 0):
        xlist[2] = xlist[1]
        flist[2] = flist[1]

    # Otherwise the root is between trials 1 and 2, set those to the limits
    else:
        xlist[0] = xlist[1]
        flist[0] = flist[1]

    # Get the new trial at the midpoint
    xlist[1] = 0.5 * (xlist[0] + xlist[2])
    flist[1] = func_freq(xlist[1])


xmax = xlist[1]

# Print wavelength of the maximum
numax = xmax * k * T / h
print("Value of x at max", xmax)
print("Trial function value at maximum", flist[1])
print("Frequency at max", numax)

# Plot the Planck function
# Make a list of wavelengths in nm and m
nu = np.linspace(0.15e14, 30.0e14, 500)

# Compute the Planck function
# dn = - dl/l^2 c
# hc^2/l^5 dl = hc^2/l^5  l^2/c dn = hc/l^3 dn = hn^3/c^2 dn
Bnu = (2 * h * nu**3 / c**2) / (np.exp(h * nu / (k * T)) - 1)

# Plot the Planck function against lambda in nm
plt.plot(nu, Bnu, '-b')

# Compute the wavlength at the max from the first part
Bnumax = (2 * h * numax**3 / c**2) / (np.exp(h * numax / (k * T)) - 1)

# Plot a a point on the
plt.plot(numax, Bnumax, 'or')

# Add some labels
plt.xlabel(r'$\nu$ (Hz)')
plt.ylabel(r'$B_\nu$ (W/m$^2$ Sr Hz)')

# Add title
plt.title(r'$B_\nu$ vs $\nu$')

plt.show()


####################
# Plot of log(Lambda*B) vs log(Lambda) for 
# various temperatures large range for lambda

# Define some constants.
k = 1.381e-23
h = 6.626e-34
c = 2.997e8

# Plot the Planck function
# Make a list of wavelengths in nm and m
lam_nm = np.logspace(1, 4, 500)
lam = lam_nm * 1e-9

# Compute and plot the Planck function at 3000 K
T = 3000
lamBlambda = lam * (2 * h * c**2 / lam**5) / \
    (np.exp(h * c / (k * T * lam)) - 1)
plt.plot(lam_nm, lamBlambda, '-r', label=r'$3000$~K')

# Compute and plot the Planck function at 5500 K
T = 5500
lamBlambda = lam * (2 * h * c**2 / lam**5) / \
    (np.exp(h * c / (k * T * lam)) - 1)
plt.plot(lam_nm, lamBlambda, '-g', label=r'$5500$~K')

# Compute and plot the Planck function at 30,000 K
T = 30000
lamBlambda = lam * (2 * h * c**2 / lam**5) / \
    (np.exp(h * c / (k * T * lam)) - 1)
plt.plot(lam_nm, lamBlambda, '-b', label=r'$3000$~K')

# Set some limits
plt.ylim((1e4, 1e11))

# Set x and y scales to logarithmic
plt.xscale('log')
plt.yscale('log')

# Add some labels
plt.xlabel(r'$\lambda$ (nm)')
plt.ylabel(r'$\lambda B_\lambda$ (W/m$^2$ Sr)')

# Add a legend
plt.legend()

# Add title
plt.title(r'Large $\lambda$ range')

plt.show()


####################
# Plot of log(Lambda*B) vs log(Lambda) for 
# various temperatures Narrow range for lambda

# Define some constants.
k = 1.381e-23
h = 6.626e-34
c = 2.997e8

# Plot the Planck function
# Make a list of wavelengths in nm and m
lam_nm = np.logspace(1, 4, 500)
lam = lam_nm * 1e-9

# Compute and plot the Planck function at 3000 K
T = 3000
lamBlambda = lam * (2 * h * c**2 / lam**5) / \
    (np.exp(h * c / (k * T * lam)) - 1)
plt.plot(lam_nm, lamBlambda, '-r', label=r'$3000$~K')

# Compute and plot the Planck function at 5500 K
T = 5500
lamBlambda = lam * (2 * h * c**2 / lam**5) / \
    (np.exp(h * c / (k * T * lam)) - 1)
plt.plot(lam_nm, lamBlambda, '-g', label=r'$5500$~K')

# Compute and plot the Planck function at 30,000 K
T = 30000
lamBlambda = lam * (2 * h * c**2 / lam**5) / \
    (np.exp(h * c / (k * T * lam)) - 1)
plt.plot(lam_nm, lamBlambda, '-b', label=r'$3000$~K')

# Set some limits
plt.xlim((400, 800))
plt.ylim((1e4, 1e11))

# Set x and y scales to logarithmic
plt.xscale('log')
plt.yscale('log')

# Add some labels
plt.xlabel(r'$\lambda$ (nm)')
plt.ylabel(r'$\lambda B_\lambda$ (W/m$^2$ Sr)')

# Add a legend
plt.legend()

# Add title
plt.title(r'Narrow $\lambda$ range')

plt.show()
