from numpy import arange, zeros
from math import exp, pow
from matplotlib.pyplot import plot, show

prob = 0.05
N = 50
DEBUG = 1


def factorial(n):
    if n == 1:
        return 1
    elif n == 0:
        return 1
    else:
        return n * factorial(n - 1)


def binomial(n, k):

    if k == 0:
        return 1
    else:
        return factorial(n) / factorial(k) / factorial(n - k)


def binomial_dist(n, k, p):
    return binomial(n, k) * (p**k) * ((1 - p)**(n - k))


def poisson(k, lam):
    x = lam**k
    print(x)
    y = factorial(int(k))
    print(y)
    return (x) * exp(-lam) / y


x = range(0, N + 1, 1)
pb = zeros(N + 1)
pp = zeros(N + 1)
lam = prob * N
for i in range(0, N + 1):
    #    pb[i] = binomial_dist(N,i,prob)
    pp[i] = poisson(i, lam)

if DEBUG:
    plot(x, pb, "ro")
    plot(x, pb, "r")
    plot(x, pp, "bo")
    plot(x, pp, "b")
    show()
