import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
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
    n = u1.shape[1]

    minmax_error = np.inf
    A_best = None

    for idx in itertools.combinations(range(0, n), 3):
        uv1 = u1[:, idx]
        uv2 = u2[:, idx]

        uv2 = np.vstack((uv2, [1, 1, 1]))
        A = uv1 @ np.linalg.inv(uv2)
        ux = A @ np.vstack((u2, np.ones(n)))

        max_error = np.linalg.norm(u1 - ux, axis=0).max()

        if max_error < minmax_error:
            minmax_error = max_error
            A_best = A

    return A_best


if __name__ == "__main__":
    image = plt.imread('daliborka_01.jpeg').copy()

    u2 = np.array([
        [ -174.5,  -220.6,  -236.5,  -354.7,  -236.1,  -360.2,  -333.5],
        [   87.8,   187.3,   205.9,   392.5,   495.3,   559.3,   742.3]
    ])

    u1 = np.array([
        [146, 273, 300, 550, 643, 745, 958],
        [185, 213, 227, 319, 153, 288, 203]
    ], dtype=float)

    # plt.imshow(image)
    # u1 = plt.ginput(7)

    colors = np.array([
        [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 0], [1, 1, 1]
    ], dtype=float)

    for (x, y), c in zip(u1.T.astype(int), colors):
        image[y, x] = c * 255

    plt.imsave('01_daliborka_points.png', image)

    A = estimate_A(u2, u1)

    ux = A @ np.vstack((u2, np.ones(u2.shape[1])))
    errors = 100 * (ux - u1)

    fig = plt.figure()
    plt.imshow(image)

    for (x, y), (e_x, e_y), c, in zip(u1.T, errors.T, colors):
        plt.plot(x, y, 'o', color=c, fillstyle='none')
        plt.plot((x, x + e_x), (y, y + e_y), 'r-')

    plt.show()
        
    fig.savefig( '01_daliborka_errs.pdf' )

    sio.savemat('01_points.mat', {'u': u1, 'A': A})
