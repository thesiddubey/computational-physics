import matplotlib.pyplot as plt
import numpy as np

####################
# Plot of Theta as a function of x for various values of n 
# from n[0->5]. O,1&5 have analytic solutions

def LaneEmden_theta(x,n) :

    # Rewrite the Lane-Emden equation as
    # df/dx = - x^2 theta^n
    # dtheta/dx = f/x^2

    theta = 0.0*x
    f = 0.0*x
    theta[0] = 1.0
    f[0] = 0.0
    for j in range(1,len(x)) :

        dfdx = -x[j-1]**2 * theta[j-1]**n
        dthetadx = f[j-1]/x[j-1]**2

        f[j] = f[j-1] + dfdx*(x[j]-x[j-1])
        theta[j] = theta[j-1] + dthetadx*(x[j]-x[j-1])

        # Stop if theta becomes negative
        if (theta[j]<=0) :
            break
    
    return theta


# Specify figure and axes so that the axis labels are right
plt.figure(figsize=(5,3.5))
plt.axes([0.15,0.15,0.8,0.8])

# Loop over n and plot lines
x=np.linspace(1e-4,10,10000)
for n in [0,1,2,3,4,5] :

    theta = LaneEmden_theta(x,n)

    plt.plot(x,theta,'-',label='n=%i'%n)


# Plot the analytical solutions
#  n=0 -- theta(x) = 1 - x^2/6
xa = np.linspace(1e-4,10,100)
theta_0 = 1.0 - xa**2/6.0
plt.plot(xa,theta_0,'--r',label='n=0 analytical',linewidth=2)

#  n=1 -- theta(x) = sin(x)/x
theta_1 = np.sin(xa)/xa
plt.plot(xa[xa<5],theta_1[xa<5],'--r',label='n=1 analytical',linewidth=2)

#  n=5 -- theta(x) = 1 / sqrt(1 + (x**2 / 3))
theta_5 = 1.0/np.sqrt(1.0 + (xa**2/3.0))
plt.plot(xa[xa<5],theta_5[xa<5],'--r',label='n=5 analytical',linewidth=2)

plt.ylim((0,1))
plt.legend(fontsize=8)

plt.xlabel(r'$x$')
plt.ylabel(r'$\theta(x)$')

plt.show()


####################
# Calculating rho_c and K for n = 3 case for a solar mass star

n=3

theta = LaneEmden_theta(x,n)

theta_interior = theta[theta>0]
x_interior = x[theta>0]
x0 = x_interior[-1]

theta_dot0 = - np.sum(theta_interior**3 * x_interior**2)*(x[1]-x[0]) / x0**2

print("x0 is",x0)
Rsun = 7e5
G = 6.67e-11

alpha = Rsun/x0
rhoc = - 1.998e30/(4.0*np.pi*alpha**3 * x0**2 * theta_dot0)
K = 4.0*np.pi*G*alpha**2/((n+1)*rhoc**((1.0-n)/n))

print("rhoc = %15.8e"%rhoc)
print("K = %15.8e"%K)

####################
# Plot of rho(density), Pressure and temperature as a function
# of r/R_star using the Polytropic equations. To calculate temperature
# we find mean molecular mass for fully ionized plasma using
# X = 0.55, Y = 0.4

theta = LaneEmden_theta(x,3)
rhoc = 7.627e4 # kg/m^3
K = 3.84e9 # SI units
gamma = 1.0+1./3.
mp = 1.67e-27
mu = mp/( 0.55*2 + 0.4*3/4.0 + 0.05*0.5 )
k = 1.381e-23

print("gamma = ",gamma)

rho = rhoc * theta**3
P = K * rho**gamma
T = P/rho * mu/k
r = x/x0

plt.figure(figsize=(8,2.5))

plt.axes([0.105,0.19,0.22,0.72])
plt.plot(r,rho,'-b')
plt.xlim((0,1))
plt.ylim((0,8e4))
plt.xlabel(r'$r/R$')
plt.ylabel(r'$\rho$ (kg/m$^3$)')


plt.axes([0.105*2+0.22,0.19,0.22,0.72])
plt.plot(r,P,'-b')
plt.xlim((0,1))
plt.ylim((0,1.4e16))
plt.xlabel(r'$r/R$')
plt.ylabel(r'$P$ (kg/m s$^2$)')


plt.axes([0.105*3+0.22*2,0.19,0.22,0.72])
plt.plot(r,T,'-b')
plt.xlim((0,1))
plt.ylim((0,1.4e7))
plt.xlabel(r'$r/R$')
plt.ylabel(r'$T$ (K)')

plt.show()

####################
# Plot of dL/dr as a function of r/R_star where dL/dr is
# due to proton-proton chain and CNO cycle

# Specify figure and axes so that the axis labels are right
plt.figure(figsize=(5,3.5))
plt.axes([0.15,0.15,0.8,0.8])

# Get the specific energy generation rates
T6 = T/1e6
T8 = T/1e8
Xh = 0.55
Xcno = 0.03*Xh
epp = 1.07e-7 * (rho/1.0e5) * Xh**2 * T6**4
ecno = 8.24e-26 * (rho/1.0e5) * Xh * Xcno * T6**19.9

dLdr = 4.0*np.pi * (r*Rsun)**2 * rho * ( epp + ecno )
dLdr_pp = 4.0*np.pi * (r*Rsun)**2 * rho * ( epp )
dLdr_cno = 4.0*np.pi * (r*Rsun)**2 * rho * ( ecno )

plt.plot(r,dLdr_pp,'--r')
plt.plot(r,dLdr_cno,'--g')
plt.plot(r,dLdr,'-b')
plt.xlim((0,1))
plt.ylim((0,2.5e12))
plt.xlabel(r'$r/R$')
plt.ylabel(r'$dL/dr$ (W/m)')

plt.show()
