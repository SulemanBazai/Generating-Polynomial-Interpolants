import numpy as np
import matplotlib.pyplot as plt

#function that creates the vandermonde matrix to model an nth order polynomial interpolant. This could be generalized further, but this code was only intended to be used for the 3rd, 5th, and 15th order polynomials.
def get_vandermonde(n,x):
    V = np.zeros((n+1)**2).reshape(n+1,n+1)
    for i in range(len(x)):
        if len(x)==4:
            V[i] = np.array([1,x[i],x[i]**2,x[i]**3])
        elif len(x)==6:
            V[i] = np.array([1,x[i],x[i]**2,x[i]**3,x[i]**4,x[i]**5])
        elif len(x)==16:
            V[i] = np.array([1,x[i],x[i]**2,x[i]**3,x[i]**4,x[i]**5,x[i]**6,x[i]**7,x[i]**8,x[i]**9,x[i]**10,x[i]**11,x[i]**12,x[i]**13,x[i]**14,x[i]**15])
    return V

# "b" is the true function with b=1/(1+25*(x**2))
#solve the system V@[1-dimensional coefficients array]=b for the coefficients of the polynomial 
def get_coefficients(n):
    x = np.linspace(-1.0,1.0,n+1)
    b = 1/(1+25*x**2)
    V = get_vandermonde(n,x)
    coefficients = np.linalg.solve(V,b)
    return coefficients

coefficients_3 = get_coefficients(3)
coefficients_5 = get_coefficients(5)
coefficients_15 = get_coefficients(15)

#we are plotting 50 evenly spaced points in the x interval (-.75,.75) for the true function, and the polynomial approximation functions
points = np.linspace(-.750,.750,50)
y = 1/(1+25*points**2) #these are the y values for the true function
y_3 = np.polyval(coefficients_3[::-1],points)
y_5 = np.polyval(coefficients_5[::-1],points)
y_15 = np.polyval(coefficients_15[::-1],points) #these are the y values for the 15th order polynomial approximation

plt.plot(points,y,label="true")
plt.plot(points,y_3,label="c3")
plt.plot(points,y_5,label="c5")
plt.plot(points,y_15,label="c15")
plt.legend(loc='upper left')
fig = plt.gca()
