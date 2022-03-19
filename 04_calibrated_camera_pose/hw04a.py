import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.io as sio
import scipy.linalg as slinalg
import itertools

from typing import Tuple


def p3p_polynom(d12, d23, d31, c12, c23, c31):
    """ 
    >>> a0, a1, a2, a3, a4 = p3p_polynom(d12, d23, d31, c12, c23, c31)
    """

    a4 = -4*d23**4*d12**2*d31**2*c23**2+d23**8-2*d23**6*d12**2-2*d23**6*d31**2+d23**4*d12**4+2*d23**4*d12**2*d31**2+d23**4*d31**4

    a3 = 8*d23**4*d12**2*d31**2*c12*c23**2+4*d23**6*d12**2*c31*c23-4*d23**4*d12**4*c31*c23+4*d23**4*d12**2*d31**2*c31*c23-4*d23**8*c12+4*d23**6*d12**2*c12+8*d23**6*d31**2*c12-4*d23**4*d12**2*d31**2*c12-4*d23**4*d31**4*c12

    a2 = -8*d23**6*d12**2*c31*c12*c23-8*d23**4*d12**2*d31**2*c31*c12*c23+4*d23**8*c12**2-4*d23**6*d12**2*c31**2-8*d23**6*d31**2*c12**2+4*d23**4*d12**4*c31**2+4*d23**4*d12**4*c23**2-4*d23**4*d12**2*d31**2*c23**2+4*d23**4*d31**4*c12**2+2*d23**8-4*d23**6*d31**2-2*d23**4*d12**4+2*d23**4*d31**4

    a1 = 8*d23**6*d12**2*c31**2*c12+4*d23**6*d12**2*c31*c23-4*d23**4*d12**4*c31*c23+4*d23**4*d12**2*d31**2*c31*c23-4*d23**8*c12-4*d23**6*d12**2*c12+8*d23**6*d31**2*c12+4*d23**4*d12**2*d31**2*c12-4*d23**4*d31**4*c12

    a0 = -4*d23**6*d12**2*c31**2+d23**8-2*d23**4*d12**2*d31**2+2*d23**6*d12**2+d23**4*d31**4+d23**4*d12**4-2*d23**6*d31**2

    return np.array([a0, a1, a2, a3, a4])


def p3p_dverify(n1, n2, n3, d12, d23, d31, c12, c23, c31):
    """
    Function p3p_dverify for verification of computed camera-to-point distances using the cosine law.
    Use this function in p3p_distances. The function returns vector of three errors, one for each equation. 
    Each computed error should be distance (not squared), relative to particular d_{jk}
    """
    error = lambda x, y, d, c: (np.sqrt(x ** 2 + y ** 2 - 2 * x * y * c) - d) / d
    e1 = error(n1, n2, d12, c12)
    e2 = error(n2, n3, d23, c23)
    e3 = error(n1, n3, d31, c31)
    return np.array([e1, e2, e3])


def p3p_distances(d12, d23, d31, c12, c23, c31):
    """
    >>> n1s, n2s, n3s = p3p_distances(d12, d23, d31, c12, c23, c31)
    """
    a0, a1, a2, a3, a4 = p3p_polynom(d12, d23, d31, c12, c23, c31)
    
    C = np.array([
        [0, 0, 0, -a0 / a4],
        [1, 0, 0, -a1 / a4],
        [0, 1, 0, -a2 / a4],
        [0, 0, 1, -a3 / a4]
    ])
    
    # solve eq. (7.88)
    n12s = np.linalg.eigvals(C)
        
    n1s = []
    n2s = []
    n3s = []
    
    threshold = 1e-4
    
    for n12 in n12s:
        # complex solutions are artifacts of the method and should not be further considered
        if np.iscomplex(n12):
            continue
        
        n12 = np.real(n12)
        
        # eqs. (7.69) - (7.74)
        m1 = d12 ** 2
        p1 = -2 * d12 ** 2 * n12 * c23
        q1 = d23 ** 2 * (1 + n12 ** 2 - 2 * n12 * c12) - d12 ** 2 * n12 ** 2
        m2 = d31 ** 2 - d23 ** 2
        p2 = 2 * d23 ** 2 * c31 - 2 * d31 ** 2 * n12 * c23
        q2 = d23 ** 2 - d31 ** 2 * n12 ** 2
        
        # eq. (7.89)
        n13 = (m1 * q2 - m2 * q1) / (m1 * p2 - m2 * p1)
        
        # eqs. (7.91) - (7.93)
        n1 = d12 / np.sqrt(1 + n12 ** 2 - 2 * n12 * c12)
        n2 = n1 * n12
        n3 = n1 * n13
        
        errors = p3p_dverify(n1, n2, n3, d12, d23, d31, c12, c23, c31)
        # print(errors)
        
        if np.all(errors <= threshold):
            n1s.append(n1)
            n2s.append(n2)
            n3s.append(n3)
            
    
    return n1s, n2s, n3s
    
    
def p3p_angles(x1, x2, x3, K):
    """
    Solves eq. (7.59)
    """
    K_inv = np.linalg.inv(K)
    
    norm = lambda x: np.linalg.norm(x)
    cos2 = lambda x, y: ((x.T @ K_inv.T @ K_inv @ y) / (norm(K_inv @ x) * norm(K_inv @ y)))[0][0]
    
    c12 = cos2(x1, x2)
    c23 = cos2(x2, x3)
    c31 = cos2(x1, x3)
    
    return c12, c23, c31
    
    
def e2p(X):
    """ 
    Euclidian to projective coordinates: (n, m) -> (n + 1, m)
    """
    return np.vstack([X, np.ones(X.shape[1])])


def p2e(X):
    """ 
    Projective to euclidian coordinates: (n, m) -> (n - 1, m)
    """
    return X[:-1] / X[-1]


if __name__ == "__main__":

    cvec = lambda x: np.asarray(x).reshape(3, 1)
    rvec = lambda x: np.asarray(x).reshape(3)
    norm = lambda x: np.linalg.norm(x)

    K = R = I = np.eye(3)
    C = cvec([1, 2, -3])
    f = 1

    P = 1 / f * K @ R @ np.c_[I, -C]

    #! Task 1.1
    X1 = cvec([0, 0, 0])
    X2 = cvec([1, 0, 0])
    X3 = cvec([0, 1, 0])
    
    # project points by P
    x1 = P @ e2p(X1)
    x2 = P @ e2p(X2)
    x3 = P @ e2p(X3)
    
    # express in β base x_β = [x_ɑ, 1]
    x1 = x1 / x1[-1]
    x2 = x2 / x2[-1]
    x3 = x3 / x3[-1]
    
    # compute the cosines
    c12, c23, c31 = p3p_angles(x1, x2, x3, K)
    
    # compute distances between points
    d12 = norm(X1 - X2)
    d23 = norm(X2 - X3)
    d31 = norm(X1 - X3)
    
    # compute the camera-points distances η
    n1s, n2s, n3s = p3p_distances(d12, d23, d31, c12, c23, c31)
    
    # compare with correct known values
    n1_true = norm(C - X1)
    n2_true = norm(C - X2)
    n3_true = norm(C - X3)
    
    print(n1s, n1_true)
    print(n2s, n2_true)
    print(n3s, n3_true)
    
    
    #! Task 1.2
    X1 = cvec([1, 0, 0])
    X2 = cvec([0, 2, 0])
    X3 = cvec([0, 0, 3])
    
    c12 = 0.9037378393
    c23 = 0.8269612542
    c31 = 0.9090648231
    
    # compute distances between points
    d12 = norm(X1 - X2)
    d23 = norm(X2 - X3)
    d31 = norm(X1 - X3)
    
    # compute the camera-points distances η
    n1s, n2s, n3s = p3p_distances(d12, d23, d31, c12, c23, c31)
    
    print(n1s, n2s, n3s)

    
    #! Task 2
    K = sio.loadmat("K.mat")["K"]
    file = sio.loadmat("daliborka_01-ux.mat")
    u = file['u']
    x = file['x']
    
    indices = np.array([58, 85, 1, 25, 98, 62, 100, 53, 34, 51]) - 1
    
    n1s_all = []
    n2s_all = []
    n3s_all = []
    
    for idx in itertools.combinations(range(0, len(indices)), 3):
        idx = np.array(idx)
        idx = indices[idx]

        X1, X2, X3 = x[:, idx].T
        
        # compute distances between points
        d12 = norm(X1 - X2)
        d23 = norm(X2 - X3)
        d31 = norm(X1 - X3)
        
        x1, x2, x3 = e2p(u[:, idx]).T
        
        x1 = cvec(x1)
        x2 = cvec(x2)
        x3 = cvec(x3)
        
        # compute the cosines
        c12, c23, c31 = p3p_angles(x1, x2, x3, K)
        
        # compute the camera-points distances η
        n1s, n2s, n3s = p3p_distances(d12, d23, d31, c12, c23, c31)
        
        n1s_all.extend(n1s)
        n2s_all.extend(n2s)
        n3s_all.extend(n3s)


    plt.xlabel('trial')
    plt.ylabel('distance [m]')
    plt.plot(n1s_all, color='red', label="$η_1$")
    plt.plot(n2s_all, color='green', label="$η_2$")
    plt.plot(n3s_all, color='blue', label="$η_3$")
    plt.ylim(0, 0.9)
    plt.legend()
    plt.savefig('04_distances.pdf')
    # plt.show()
