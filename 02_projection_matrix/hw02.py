import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.io as sio
import scipy.linalg as slinalg
import itertools


def estimate_A(u2: np.array, u1: np.array):
    """ Estimation of the affine transformation from u2 to u1 from 3 point correspondences

    Parameters
    ----------
    u2 : np.array
        2 x n matrix
    u1 : np.array
        2 x n matrix
    """

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


if __name__ == "__main__":
    image = plt.imread('daliborka_01.jpeg').copy()

    indices = np.array([25, 21, 48, 73, 27, 26, 51, 2, 37, 17]) - 1

    file = sio.loadmat('daliborka_01-ux.mat')
    u = file['u']
    x = file['x']


    Q_best, points_selected, errors_max, errors_vectors, Q_all = estimate_Q(u, x, indices)

    Xh = np.r_[x, [np.ones(x.shape[1])]]
    Ur = Q_best @ Xh
    Ur = Ur[:2] / Ur[-1]


    fig, ax = plt.subplots(1, 1)
    ax.imshow(image)
    ax.scatter(u[0], u[1], c='b', marker='.', label='Orig. pts')
    ax.scatter(u[0, points_selected], u[1, points_selected], c='y', marker='.', label='Used for Q')
    ax.scatter(Ur[0], Ur[1], marker='o', label='Reprojected', facecolors='none', edgecolors='r')
    ax.set_title('Original and reprojected points')
    ax.set_xlabel('x [px]')
    ax.set_ylabel('y [px]')
    plt.legend()
    plt.savefig('02_Q_projections.pdf')
    # plt.show()
    plt.close()

    fig, ax = plt.subplots(1, 1)
    ax.imshow(image)
    ax.scatter(u[0], u[1], c='b', marker='.', label='Orig. pts')
    ax.scatter(u[0, points_selected], u[1, points_selected], c='y', marker='.', label='Used for Q')
    UrE = Ur + 100 * errors_vectors
    ax.plot([Ur[0], UrE[0]], [Ur[1], UrE[1]], 'r-')
    ax.plot([Ur[0, 0], UrE[0, 0]], [Ur[1, 0], UrE[1, 0]], 'r-', label='Errors (100x)')
    ax.set_title('Reprojection errors (100x enlarged)')
    ax.set_xlabel('x [px]')
    ax.set_ylabel('y [px]')
    plt.legend()
    plt.savefig('02_Q_projections_errors.pdf')
    # plt.show()
    plt.close()

    fig, ax = plt.subplots(1, 1)
    ax.plot(np.log10(errors_max))
    ax.set_title('Maximal reproj. err. for each tested Q')
    ax.set_xlabel('selection index')
    ax.set_ylabel('$\log_{10}$ of max. err. [px]')
    plt.savefig('02_Q_maxerr.pdf')
    # plt.show()
    plt.close()

    
    fig, ax = plt.subplots(1, 1)
    errors = np.linalg.norm(errors_vectors, axis=0)
    ax.plot(errors)
    ax.set_title('Maximal reproj. err. for each tested Q')
    ax.set_xlabel('point index')
    ax.set_ylabel('reproj. err. [px]')
    plt.savefig('02_Q_pointerr.pdf')
    # plt.show()
    plt.close()