import numpy as np
from numpy.linalg import norm, inv
import numpy as np
import itertools
import matplotlib.pyplot as plt
import scipy.io as sio


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
    Computes η_1, η_2, η_3
    >>> n1s, n2s, n3s = p3p_distances(d12, d23, d31, c12, c23, c31)
    Returns 3-tuple of η arrays
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
    x1 = cvec(x1)
    x2 = cvec(x2)
    x3 = cvec(x3)
    
    K_inv = np.linalg.inv(K)
    
    cos2 = lambda x, y: ((x.T @ K_inv.T @ K_inv @ y) / (norm(K_inv @ x) * norm(K_inv @ y)))[0][0]
    
    c12 = cos2(x1, x2)
    c23 = cos2(x2, x3)
    c31 = cos2(x1, x3)
    
    return c12, c23, c31


def p3p_RC(N, u, X, K):
    """
    Computes calibrated camera centre C and orientation R from three scene-to-image 
    correspondences (X, u), using already computed distances N = [η_1, η_2, η_3].
    The function takes one configuration of η_i and returns a single R and C.
    Note that R must be ortho-normal with determinant equal to +1.
    """
    # x_ɑ -> x_β
    u = e2p(u)
    
    n1, n2, n3 = np.transpose(N)
    x1, x2, x3 = np.transpose(u)
    X1, X2, X3 = np.transpose(X)
    
    K_inv = inv(K)
    
    # x_β -> x_γ
    x1 = K_inv @ x1
    x2 = K_inv @ x2
    x3 = K_inv @ x3
    
    # eq. (7.121)
    Y1 = n1 * x1 / norm(x1)
    Y2 = n2 * x2 / norm(x2)
    Y3 = n3 * x3 / norm(x3)
    
    # eq. (7.130) - (7.132)
    Z3e = Y3 - Y1
    Z2e = Y2 - Y1
    Z1e = np.cross(Z2e, Z3e)
    Z3d = X3 - X1
    Z2d = X2 - X1
    Z1d = np.cross(Z2d, Z3d)
    
    # eq. (7.134)
    R = np.c_[Z1e, Z2e, Z3e] @ inv(np.c_[Z1d, Z2d, Z3d])
    
    # determinant equals to 1
    # print(np.linalg.det(R))
    
    # orthonormal basis
    # print(np.allclose(R.T @ R - np.eye(3), 0, 1e-8))
    
    # eq. (7.135)
    C = cvec(X1 - R.T @ Y1)
    
    return R, C
    
    
def e2p(X):
    """ 
    Euclidian to projective coordinates: (n, m) -> (n + 1, m)
    """
    X = np.asarray(X)
    return np.vstack([X, np.ones(X.shape[1])])


def p2e(X):
    """ 
    Projective to euclidian coordinates: (n, m) -> (n - 1, m)
    """
    X = np.asarray(X)
    return X[:-1] / X[-1]


def cvec(x):
    return np.asarray(x).reshape(-1, 1)


def rvec(x):
    return np.asarray(x).reshape(-1)


def plot_csystem(ax, base, origin, name, color):
    """ 
    >>> plot_csystem(ax, np.eye(3), np.zeros([3, 1]), 'k', 'd')
    """
    C = origin[:, 0]
    Cx, Cy = base[:, 0], base[:, 1]
    Cz = None if base.shape[1] == 2 else base[:, 2]
    
    # Cx /= np.linalg.norm(Cx)
    # Cy /= np.linalg.norm(Cy)
    
    ax.quiver3D(*C, *Cx, arrow_length_ratio=0.1, color=color)
    ax.quiver3D(*C, *Cy, arrow_length_ratio=0.1, color=color)
    
    ax.text(*(C + Cx), f'${name}_x$')
    ax.text(*(C + Cy), f'${name}_y$')
    
    if Cz is not None:
        # Cz /= np.linalg.norm(Cz)
        ax.quiver3D(*C, *Cz, arrow_length_ratio=0.1, color=color)
        ax.text(*(C + Cz), f'${name}_z$')


def u2H(u, u0):
    """
    Create function u2H computing homography from four image matches.
    Let u be the image coordinates of points in the first image (2x4 matrix, np.array) 
    and u0 (2x4) be the image coordinates of the corresponding points in the second image. 
    Then H is a 3x3 homography matrix (np.array), such that
    """
    U1, V1 = np.asarray(u)
    U2, V2 = np.asarray(u0)

    ones, zeros = np.ones_like(U1), np.zeros_like(U1)
    
    A = np.r_[
        np.c_[U1, V1, ones, zeros, zeros, zeros, -U2 * U1, -U2 * V1, -U2],
        np.c_[zeros, zeros, zeros, U1, V1, ones, -V2 * U1, -V2 * V1, -V2]
    ]

    U, S, V = np.linalg.svd(A)
    H = V[-1].reshape(3, 3)

    if H[2, 2] != 0:
        H = H / H[2, 2]
    else:
        print('H[2, 2] == 0')

    return H


def dist(H, u, u0):
    """ Function, calculates one-way reprojection error

    Parameters
    ----------
    H : np.array (3, 3)
        Homography matrix
    u : np.array (2, n)
        Image coordinates of points in the first image
    u0 : np.array (2, n)
        Image coordinates of points in the second image

    Returns
    -------
    np.array (N)
        Per-correspondence Eucledian error
    """
    return norm(u0 - p2e(H @ e2p(u)), axis=0)


def u2h_optim(u, u0):
    u = np.asarray(u)
    u0 = np.asarray(u0)
    
    indices = np.arange(u.shape[1])
    min_transfer_error = np.inf
    
    for idx in itertools.combinations(indices, 4):
        
        H = u2H(u[:, idx], u0[:, idx])
        
        errors = dist(H, u, u0)
        
        transfer_error = errors.max()
        
        if transfer_error < min_transfer_error:
            min_transfer_error = transfer_error
            print(min_transfer_error)
            H_best = H
            idx_best = idx
        
    return H_best, idx_best
