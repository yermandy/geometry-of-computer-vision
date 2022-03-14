import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.io as sio
import scipy.linalg as slinalg
import itertools

from typing import Tuple


def estimate_Q(U, X, indices):
    Q_best = None
    Q_all = []
    errors_max = []
    errors_vectors = []
    minmax_error = np.inf

    # points in homogeneous coordinate system
    Xh = np.r_[X, [np.ones(X.shape[1])]]

    ones = np.ones(6)
    zeros = np.zeros(6)

    for idx in itertools.combinations(range(0, len(indices)), 6):
        idx = np.array(idx)
        idx = indices[idx]

        x, y, z = X[:, idx]
        u, v = U[:, idx]

        # construct M matrix
        M = np.r_[
            np.c_[x, y, z, ones, zeros, zeros, zeros, zeros, -u * x, -u * y, -u * z, -u],
            np.c_[zeros, zeros, zeros, zeros, x, y, z, ones, -v * x, -v * y, -v * z, -v]
        ]

        for i in itertools.combinations(range(0, 12), 11):
            i = np.array(i)
            
            # rank deficient matrix
            M11 = M[i]
            Q = scipy.linalg.null_space(M11).reshape(3, 4)
            Q_all.append(Q)

            # preproject points
            Ur = Q @ Xh
            Ur = Ur[:2] / Ur[-1]

            # calculate reprojection errors
            error_vectors = U - Ur
            errors = np.linalg.norm(error_vectors, axis=0)
            max_error = errors.max()
            errors_max.append(max_error)

            if max_error < minmax_error:
                minmax_error = max_error
                errors_vectors = error_vectors
                Q_best = Q
                points_selected = idx

    return Q_best, points_selected, errors_max, errors_vectors, Q_all


def Q2KRC(Q: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ Decomposes Projection matrix Q (3x4) to:
        calibration matrix K (3x3), rotation matrix R (3x3),
        projection center C (3x1) such that Q = λ [K R | - K R C],
        where K(3,3) = 1, K(1,1) > 0, and det(R) = 1. 
    """
    M = Q[:, :-1]
    m = Q[:, -1]
    norm = np.sum(M[2] ** 2)
    K = np.zeros((3, 3))
    K[1, 2] = M[1] @ M[2] / norm
    K[0, 2] = M[0] @ M[2] / norm
    K[1, 1] = np.sqrt(M[1] @ M[1] / norm - K[1, 2] ** 2)
    K[0, 1] = (M[0] @ M[1] / norm - K[0, 2] * K[1, 2]) / K[1, 1]
    K[0, 0] = np.sqrt(M[0] @ M[0] / norm - K[0, 1] ** 2 - K[0, 2] ** 2)
    K[2, 2] = 1
    R = np.sign(np.linalg.det(M)) / np.linalg.norm(M[2]) * np.linalg.inv(K) @ M
    C = -np.linalg.inv(M) @ m
    C = C.reshape(3, -1)
    return K, R, C
    

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
    
    
if __name__ == "__main__":
    image = plt.imread('daliborka_01.jpeg').copy()

    indices = np.array([25, 21, 48, 73, 27, 26, 51, 2, 37, 17]) - 1

    file = sio.loadmat('daliborka_01-ux.mat')
    u = file['u']
    x = file['x']

    Q_best, points_selected, errors_max, errors_vectors, Q_all = estimate_Q(u, x, indices)
    
    K, R, C = Q2KRC(Q_best)
    I = np.eye(3)
    R_inv = np.linalg.inv(R)
    K_inv = np.linalg.inv(K)
    
    # horizontal pixel size (b1 norm) is 5 μm
    b1_norm = 5 * 10 ** -6
    # see eq. (7.16)
    f = K[0, 0] * b1_norm
    # see eq. (7.8)
    A = 1 / f * K @ R
    # see eq. (7.30)
    Pb = A @ np.c_[I, -C]

    Delta = np.eye(3)
    d = np.zeros(3).reshape(3, 1)
    
    Epsilon = Delta @ R_inv
    e = C.copy()
    
    Kappa = Delta * f
    k = d.copy()
    
    Nu = Epsilon @ K_inv
    n = C.copy()
        
    Gamma = f * Delta @ R_inv
    g = C.copy()
    
    Beta = Gamma @ K_inv
    b = C.copy()
    
    Alpha = Beta @ [[1, 0], [0, 1], [0, 0]]
    a = C + Beta[:, 2].reshape(3, 1)
    
    sio.savemat('03_bases.mat', {
        'Pb': Pb, 'f': f,
        'Alpha': Alpha, 'a': a,
        'Beta': Beta, 'b': b,
        'Gamma': Gamma, 'g': g,
        'Delta': Delta, 'd': d,
        'Epsilon': Epsilon, 'e': e,
        'Kappa': Kappa, 'k': k,
        'Nu': Nu, 'n': n
    })

    
    Alpha = Alpha @ [[1100, 0], [0, 850]]
    Beta = Beta @ [[1100, 0, 0], [0, 850, 0], [0, 0, 1]]
    
    # 03_figure1.pdf
    ax = plt.axes(projection='3d')
    margin = 1
    ax.axes.set_xlim3d(-margin, margin)
    ax.axes.set_ylim3d(-margin, margin)
    ax.axes.set_zlim3d(-margin, margin)
    ax.scatter(x[0], x[1], x[2], c='b', marker='.')
    plot_csystem(ax, Delta, d, 'δ', 'black')
    plot_csystem(ax, Epsilon, e, 'ε', 'magenta')
    plot_csystem(ax, Kappa, k, 'κ', 'brown')
    plot_csystem(ax, Nu, n, 'υ', 'cyan')
    plot_csystem(ax, Beta * 50, b, 'β', 'red')
    plt.savefig('03_figure1.pdf')
    # plt.savefig('03_figure1.png', dpi=300)
    plt.close()

    # 03_figure2.pdf
    ax = plt.axes(projection='3d')
    margin = 1    
    plot_csystem(ax, Alpha, a, 'α', 'green')
    plot_csystem(ax, Beta, b, 'β', 'red')
    plot_csystem(ax, Gamma, g, 'γ', 'blue')
    plt.savefig('03_figure2.pdf')
    # plt.savefig('03_figure2.png', dpi=300)
    # plt.show()
    plt.close()
    # exit()
    
    # 03_figure3.pdf
    ax = plt.axes(projection='3d')
    margin = 1
    ax.axes.set_xlim3d(-margin, margin)
    ax.axes.set_ylim3d(-margin, margin)
    ax.axes.set_zlim3d(-margin, margin)
    ax.scatter(x[0], x[1], x[2], c='b', marker='.')
    plot_csystem(ax, Delta, d, 'δ', 'black')
    plot_csystem(ax, Epsilon, e, 'ε', 'magenta')
    for Q in Q_all:
        K, R, C = Q2KRC(Q)
        ax.scatter(*C, c='r', marker='.')
    plt.savefig('03_figure3.pdf')
    # plt.savefig('03_figure3.png', dpi=300)
    # plt.show()