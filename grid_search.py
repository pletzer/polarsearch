import numpy as np
from find_param import findParam

def gridSearch(xi0: np.array, nr1: int=11, nt1: int=21, rmin: float=0.001, rmax: float=1.0, tol: float=1.e-4):
    """
    Given an initial position in parameter space, return an array of number of iterations for each 
    of the grid target points
    :param xi0: initial guess in parameter space
    :param nr1: number of radial points
    :param nt1: number of poloidal points
    :param rmin: min radius
    :param rmax: max radius
    :param tol: search tolerance
    :returns arrays of num iterations for each of the target points
    """
    num_iters = np.empty((nr1, nt1), np.int32)
    xis = np.empty((nr1, nt1, 2), np.float64)
    dr = (rmax - rmin)/(nr1 - 1)
    dt = 2*np.pi/(nt1 - 1)
    x = np.empty((2,), np.float64)
    for j in range(nr1):
        rho = rmin + j*dr
        for i in range(nt1):
            the = i*dt
            x[:] = rho*np.cos(the), rho*np.sin(the)
            sol = findParam(xi0, x, tol)
            num_iters[j, i] = sol['num_iters']
            xis[j, i] = sol['xi']
            #print(f'j={j} i={i} rho={rho} the={the} x={x} num its={res[j,i]}')
    return num_iters, xis

if __name__ == '__main__':
    xi0 = np.array([1.0, 0.2])
    res = gridSearch(xi0, nr1=3, nt1=5, rmin=0.01, rmax=1.0, tol=1.e-10)
    print(res[0])
    print(res[1])
