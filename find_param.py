import numpy as np

def findParam(xi0: np.array, target_pos: np.array, tol: float=1.e-4) -> dict:
    """
    Find the root xi of target_pos(xi)
    :param xi0: initial xi guess
    :param target_pos: target position
    :param tol: tolerance
    :returns parameter and diagnostics
    """
    res = {
        'num_iters': 0,
        'xi': np.empty((2,), np.float64),
        'error': float('inf')
    }
    xi = xi0.copy()
    rho, the = xi
    x = np.array([rho*np.cos(the), rho*np.sin(the)])
    dx = target_pos - x
    error = np.sqrt(dx.dot(dx))
    while error > tol:
        rho, the = xi
        cost, sint = np.cos(the), np.sin(the)
        dx = target_pos - np.array([rho*cost, rho*sint])
        xi += np.array([(+ cost*dx[0] + sint*dx[1]),
                        (- sint*dx[0] + cost*dx[1])/rho])
        error = np.sqrt(dx.dot(dx))
        res['num_iters'] += 1
    res['xi'][:] = xi
    res['error'] = error
    return res

if __name__ == '__main__':
    # test
    xi0 = np.array([1.0, 0.2])
    target_pos = np.array([0.0, 1.0])
    res = findParam(xi0, target_pos, tol=1.e-10)
    print(f'res = {res}')
    assert(res["num_iters"] <= 10)
    rho, the = res['xi']
    xfinal = np.array([rho*np.cos(the), rho*np.sin(the)])
    dx = xfinal - target_pos
    assert(np.sqrt(dx.dot(dx)) < 1.e-10)

