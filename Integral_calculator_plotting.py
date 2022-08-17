from numpy import arange, sin, pi, cos, zeros, max, min
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle

#define the function to approximate
def f_of_x(x):
    return cos(x*x*x)/(10+x*x)

fll = -2.0
ful = +2.0

#set global ylimits
x0 = arange(fll,ful,0.01)
y0 = f_of_x(x0)

plt.ylim (1.1*min(y0), 1.1*max(y0))

def showbox(ll,rl):

#define the midpoint
    mp = (ll + rl)/2.0
#set the x and y values
    x = arange(ll,rl,0.01)
    y = f_of_x(x)
#plot the function over this range
    plt.plot(x,y,color = 'b')
#plot the x-axis
    plt.plot([1.1*ll,1.1*rl],[0,0],color='k')
#overplot a box
    currentAxis = plt.gca()
    currentAxis.add_patch(Rectangle((ll,0), (rl-ll), f_of_x(mp), color = 'g', fill = True, alpha = 0.5))


def showallboxes(fll,ful,n):
    x = zeros(n+1)
    for i in range(0,n+1):
        x[i] = fll + i*(ful-fll)/float(n)
    for i in range(0,n):
        showbox(x[i],x[i+1])
    plt.show()


#global plot limits

showallboxes(fll,ful,1)
showallboxes(fll,ful,2)
showallboxes(fll,ful,4)
showallboxes(fll,ful,8)
showallboxes(fll,ful,16)
showallboxes(fll,ful,32)
showallboxes(fll,ful,64)

