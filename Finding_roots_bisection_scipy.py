import numpy as np  
import matplotlib.pyplot as plt

def bisection(xmin,xmax,f) :

    x1 = xmin
    f1 = f(x1)
    x3 = xmax
    f3 = f(x3)

    # Take 10 iteration steps
    for i in range(50) :
        x2 = 0.5*(x1+x3)
        f2 = f(x2)

        if (f1*f2<=0) :
            x3 = x2
            f3 = f2
        else :
            x1 = x2
            f1 = f2

    return x2

# Random Funciton
def my_func(x) :

    return ( np.sinh(x)-np.exp(-x) )

# Get a root
root = bisection(-10,10,my_func)

# Print the answer
print("Root: ",root," where fucntion value is: ",my_func(root))

# Plot the answer to see what it looks like
plt.figure(figsize=(4,3))
x=np.linspace(-1,2,500)
plt.axhline(0,color='r') # Put a red line at y=0
plt.plot(x,my_func(x),'-b',label='') # Solid blue line
plt.plot(root,my_func(root),'og',label='bisection root') # Green circular point root
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
plt.tight_layout()
plt.show()


#####
# Now use the Scipy optimize root finder
import scipy.optimize as so
rf = so.root(my_func,0.0)
new_root = rf.x[0]

# Print the answer
print("SciPy root: ",new_root," where function value is",my_func(new_root))


# Plot the answer again 
plt.figure(figsize=(4,3))
x=np.linspace(-1,2,500)
plt.axhline(0,color='r') # Put a red line at y=0
plt.plot(x,my_func(x),'-b',label='') # Solid blue line
plt.plot(root,my_func(root),'og',label='bisection root') # Green circular point at our root
plt.plot(new_root,my_func(new_root),'yx',label='SciPy root') # Yellow x at the SciPy root
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
plt.tight_layout()
plt.savefig('root_comparison.png',dpi=300)
plt.show()



