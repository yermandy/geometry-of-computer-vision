from lib import *

import matplotlib.pyplot as plt
import scipy
import scipy.io as sio
import scipy.linalg as slinalg
import itertools


if __name__ == "__main__":

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
