import numpy as np
from scipy.optimize import minimize

def arg_min_function(y, y_star):
    func=np.linalg.norm(y[0:2]-y_star[0:2])+np.linalg.norm(y[2:]-y_star[2:])
    return func

def arg_min(U, y_star):
    cons=({'type':'eq', 'fun': lambda y: y[0:2]+0.1*y[2:]-U})
    y0=np.zeros(len(y_star))
    res=minimize(arg_min_function, y0, args=(y_star), constraints=cons)
    print(res.message)
    return res.x

if __name__=='__main__':
    # test for arguement minimization
    U=np.array([1, 2])
    y_star=np.array([1, 2, 3, 4])
    X=arg_min(U, y_star)
    print(X)