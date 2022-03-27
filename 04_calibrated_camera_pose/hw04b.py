from lib import *

import matplotlib.pyplot as plt
import scipy
import scipy.io as sio
import itertools


def is_orthonormal(A, th=1e-8):
    return np.allclose(A.T @ A - np.eye(3), 0, th)


def p3p_RC(N, u, X, K):
    """
    The function takes one configuration of η_i and returns a single R and C.
    Note that R must be orthonormal with determinant equal to +1.
    :param N: [η_1, η_2, η_3]
    :param u: coordinates of image points
    :param X: coordinates of world points
    :param K: Camera calibration mat
    :return: calibrated camera centre C and orientation R
    """

    # preparing data because of incorrect input (named as matlab form)...
    u = u.T
    X = X.T

    R, C = list(), list()
    new_u = list()
    inv_K = np.linalg.inv(K)
    for each in u:
            tmp = each
            if min(tmp.shape) == 2:
                tmp = list(tmp.reshape(2, ))
                tmp.append(1)
                tmp = inv_K @ tmp
            else:
                tmp = list(tmp.reshape(3, ))
                tmp = inv_K @ tmp
            new_u.append(tmp)

    u = new_u
    Y = [N[i] * (u[i] / np.linalg.norm(u[i])) for i in range(3)]

    Z2e = (Y[1] - Y[0]).reshape(3, )
    Z2d = (X[1] - X[0]).reshape(3, )
    Z3e = (Y[2] - Y[0]).reshape(3, )
    Z3d = (X[2] - X[0]).reshape(3, )

    Z1e = np.cross(Z2e, Z3e)
    Z1d = np.cross(Z2d, Z3d)

    R = np.c_[Z1e, Z2e, Z3e] @ np.linalg.inv(np.c_[Z1d, Z2d, Z3d])
    # print(R)
    # print(R.T @ R)
    # print(np.linalg.det(R))
    # print(round(np.linalg.det(R) - 1, 2))

    C = X[0].reshape(3, ) - (R.T @ Y[0].reshape(3, ))
    C = C.reshape(3, 1)

    return R, C


def find_optimal_RC(u, X, K, indices):
    X_p = e2p(X)
    u_p = e2p(u)
    
    errors_max = []
    centers_all = []
    R_best = None
    C_best = None
    idx_best = None
    error_vectors_best = None
    
    minmax_error = np.inf
    
    for idx in itertools.combinations(indices, 3):
        # sample three scene-to-image correspondences (X, u)
        X1, X2, X3 = X[:, idx].T
        u1, u2, u3 = u_p[:, idx].T
        
        # compute the cosines
        c12, c23, c31 = p3p_angles(u1, u2, u3, K)
        
        # compute distances between points
        d12 = norm(X1 - X2)
        d23 = norm(X2 - X3)
        d31 = norm(X1 - X3)
        
        # compute the camera-points distances η
        n1s, n2s, n3s = p3p_distances(d12, d23, d31, c12, c23, c31)
    
        _u = u[:, idx]
        _X = X[:, idx]
        
        for N in zip(n1s, n2s, n3s):
            R, C = p3p_RC(N, _u, _X, K)
            
            # if not is_orthonormal(R):
            #     continue
            
            Q = K @ R @ np.c_[np.eye(3), -C]
            
            # reproject points
            u_r = p2e(Q @ X_p)
            
            # calculate reprojection errors
            error_vectors = u_r - u
            errors = np.linalg.norm(error_vectors, axis=0)
            max_error = errors.max()
            errors_max.append(max_error)
            centers_all.append(C)
            
            if max_error < minmax_error:
                minmax_error = max_error
                R_best = R
                C_best = C
                idx_best = idx
                error_vectors_best = error_vectors
                print(max_error)
    
    return R_best, C_best, idx_best, errors_max, error_vectors_best, centers_all


if __name__ == "__main__":
    #! 1
    C = cvec([1, 2, -3])
    f = 1
    K = R = I = np.eye(3)
    
    P = 1 / f * K @ R @ np.c_[I, -C]
    
    # define three 3d points
    X1 = cvec([0, 0, 0])
    X2 = cvec([1, 0, 0])
    X3 = cvec([0, 1, 0])
    
    # project points by P: δ -> β
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
    
    u1 = p2e(x1)
    u2 = p2e(x2)
    u3 = p2e(x3)
    
    u = np.c_[u1, u2, u3]
    X = np.c_[X1, X2, X3]
    
    for N in zip(n1s, n2s, n3s):
        R_pred, C_pred = p3p_RC(N, u, X, K)
        
        print('sum of abs diffs')
        print(np.abs(R - R_pred).sum())
        print(np.abs(C - C_pred).sum())
        print()
    
    #! 2, 3, 4, 5
    
    K = sio.loadmat("K.mat")["K"]
    file = sio.loadmat("daliborka_01-ux.mat")
    u = file['u']
    X = file['x']
    
    indices = np.array([58, 85, 1, 25, 98, 62, 100, 53, 34, 51]) - 1
    
    R_best, C_best, idx_best, errors_max, error_vectors, centers_all = find_optimal_RC(u, X, K, indices)
    
    Q_best = K @ R_best @ np.c_[I, -C_best]
    
    # reproject points
    u_p = Q_best @ e2p(X)
    u_e = p2e(u_p)
    error_vectors = u_e - u
    
    #! 6
    # Export the optimal R, C, and point_sel (indices [i1, i2, i3] of the three points used for computing the optimal R, C) as 04_p3p.mat.
    sio.savemat('04_p3p.mat', {'R': R_best, 'C': C_best, 'point_sel': idx_best})
    
    #! 7
    # Display the image (daliborka_01) and draw u as blue dots, 
    # highlight the three points used for computing the best R, C by drawing them as yellow dots, 
    # and draw the displacements of reprojected points x multiplied 100 times as red lines. 
    # Export as 04_RC_projections_errors.pdf.
    image = plt.imread('daliborka_01.jpeg').copy()
    
    u_err = u + 100 * error_vectors
    
    fig, ax = plt.subplots(1, 1)
    ax.imshow(image)
    ax.scatter(u[0, idx_best], u[1, idx_best], marker='o', c='yellow', label='Used for P')
    ax.scatter(u[0], u[1], marker='.', c='blue', label='Orig. pts')
    # ax.scatter(u_e[0], u_e[1], marker='.', c='magenta', label='Repr. pts')
    ax.plot([u[0], u_err[0]], [u[1], u_err[1]], 'r-')
    ax.plot([u[0, 0], u_err[0, 0]], [u[1, 0], u_err[1, 0]], 'r-', label='Errors')
    plt.legend()
    plt.savefig('04_RC_projections_errors.pdf')
    # plt.show()
    plt.close()
    
    #! 8
    # Plot the decadic logarithm log10() of the maximum reprojection error of all the computed poses as 
    # the function of their trial index and export as 04_RC_maxerr.pdf. 
    # Plot the errors as points, not lines, in this case.
    fig, ax = plt.subplots(1, 1)
    ax.scatter(np.arange(len(errors_max)), errors_max, marker='.')
    ax.set_title('Maximal err. for each tested P')
    ax.set_ylabel('log10 max err [px]')
    ax.set_xlabel('trial')
    ax.set_yscale('log')
    plt.savefig('04_RC_maxerr.pdf')
    # plt.show()
    plt.close()
    
    #! 9
    # Plot the reprojection error of the best R, C on all 109 points as the function of point index and export as 04_RC_pointerr.pdf.
    errors = np.linalg.norm(error_vectors, axis=0)
    fig, ax = plt.subplots(1, 1)
    ax.plot(errors)
    ax.set_title('All point errors for the best P')
    ax.set_ylabel('err [px]')
    ax.set_xlabel('point')
    plt.savefig('04_RC_pointerr.pdf')
    # plt.show()
    plt.close()
    
    #! 10
    # Draw the coordinate systems δ (black), ε (magenta) of the optimal R, C, draw the 3D scene points (blue), 
    # and draw centers (red) of all cameras you have tested. Export as 04_scene.pdf.
    centers_all = np.hstack(centers_all)
    
    # δ base
    Delta = np.eye(3)
    d = cvec([0, 0, 0])
    
    # ε base
    Epsilon = Delta @ inv(R_best)
    e = C_best.copy()
    
    margin = 1
    
    ax = plt.axes(projection='3d')
    ax.axes.set_xlim3d(-margin, margin)
    ax.axes.set_ylim3d(-margin, margin)
    ax.axes.set_zlim3d(-margin, margin)
    ax.scatter(X[0], X[1], X[2], c='b', marker='.')
    ax.scatter(centers_all[0], centers_all[1], centers_all[2], c='r', marker='.')
    plot_csystem(ax, Epsilon, e, 'ε', 'magenta')
    plot_csystem(ax, Delta, d, 'δ', 'black')
    plt.savefig('04_scene.pdf')
    # plt.show()
    plt.close()