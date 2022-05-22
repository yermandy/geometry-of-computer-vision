from matplotlib import markers
from lib import *


def Pu2X(P1, P2, u1p, u2p):
    assert P1.shape == P2.shape == (3, 4)
    assert u1p.shape == u2p.shape

    _, n_samples = u1p.shape

    p11, p12, p13 = P1
    p21, p22, p23 = P2

    X = []

    for i in range(n_samples):
        u1i, u2i = u1p[:, i], u2p[:, i]
        u1, v1, _ = u1i
        u2, v2, _ = u2i

        D = np.array([
            u1 * p13 - p11,
            v1 * p13 - p12,
            u2 * p23 - p21,
            v2 * p23 - p22
        ])

        # TODO numerical conditioning
        # print(np.max(D) - np.min(D))

        _, _, V_t = np.linalg.svd(D)

        x = V_t[-1]
        X.append(x)

    X = np.transpose(X)
    return X


def Eu2RC(E, u1p, u2p):
    # Essential matrix decomposition with cheirality constraint: 
    #   all 3D points are in front of both cameras
    #   see 9.6.3 in book
    assert E.shape == (3, 3)
    assert u1p.shape[0] == 3
    assert u2p.shape[0] == 3

    U, D, V_t = np.linalg.svd(E)
    
    R = np.eye(3)
    C = np.zeros(3)

    P1 = np.c_[R, C]
    
    R_best = None
    C_best = None
    
    in_front_of_both_best = -np.inf
    
    for a in [-1, 1]:
        for b in [-1, 1]:
            W = np.array([
                [0,  a, 0],
                [-a, 0, 0],
                [0,  0, 1]
            ])
            
            R = U @ W @ V_t
            C = b * V_t[-1]

            P2 = np.c_[R, -R @ C]

            X = Pu2X(P1, P2, u1p, u2p)
            X = p2e(X)
                                    
            in_front_of_C1 = np.einsum('ij, ij -> j', u1p, X) > 0
            in_front_of_C2 = np.einsum('ij, ij -> j', u2p, X) > 0
            in_front_of_both = (in_front_of_C1 & in_front_of_C2).sum()
                        
            if in_front_of_both > in_front_of_both_best:
                in_front_of_both_best = in_front_of_both
                R_best = R
                C_best = C

    return R_best, C_best



def plot_3D_points(X, edges):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.set_box_aspect(aspect = (1,1,1))

    ax.set_xlim3d(-0.4, 0.4)
    ax.set_ylim3d(-0.4, 0.4)
    ax.set_zlim3d(1.5, 1.5+0.8)
    
    ax.scatter(X[0], X[1], X[2], c='r', marker='o')
    
    for x, y, z in zip(X[0, edges].T, X[1, edges].T, X[2, edges].T):
        ax.plot(x, y, z, 'y-')
    
    plt.show()
    plt.close()



if __name__ == '__main__':
    #! Part B: Cameras and reconstruction
    
    K = sio.loadmat('K.mat')['K']
    K_inv = inv(K)
    
    file = sio.loadmat('daliborka_01_23-uu.mat')
    edges = file['edges'] - 1
    u1 = file['u01']
    u2 = file['u23']
    ix = file['ix'][0] - 1
    
    image1 = plt.imread('daliborka_01.jpg')
    image2 = plt.imread('daliborka_23.jpg')

    #! 1.
    # Decompose the best E into relative rotation R and translation C (four solutions). 
    # Choose such a solution that reconstructs (most of) the points in front of both (computed) cameras.
    Fe, E, indices_best = uK2FeE_optimal(u1, u2, ix, K)
    
    u1p = K_inv @ e2p(u1)
    u2p = K_inv @ e2p(u2)
    
    R, C = Eu2RC(E, u1p, u2p)
    
    #! 2.
    # Construct projective matrices P1, P2 (including K).
    P1 = K @ np.c_[np.eye(3), np.zeros(3)]
    P2 = K @ R @ np.c_[np.eye(3), -C]
    
    #! 3.
    # Compute scene points X.
    X = Pu2X(P1, P2, e2p(u1), e2p(u2))
    X = p2e(X)
        
    #! 4.
    # Display the images, draw the input points as blue dots and the scene points X projected by appropriate P_i as red circles. 
    # Draw also the edges, connecting the original points as yellow lines. 
    # Export as 09_reprojection.pdf.
            
    # reproject points
    u1r = p2e(P1 @ e2p(X))
    u2r = p2e(P2 @ e2p(X))
    
    fig, axes = plt.subplots(1, 2)
    ax1, ax2 = axes
    
    # show images
    ax1.imshow(image1)
    ax2.imshow(image2)

    # plot edges
    for x, y in zip(u1[0, edges].T, u1[1, edges].T):
        ax1.plot(x, y, 'y-')
        
    for x, y in zip(u2[0, edges].T, u2[1, edges].T):
        ax2.plot(x, y, 'y-')

    # plot reprojected points
    ax1.scatter(u1r[0], u1r[1], c='r', marker='o')
    ax2.scatter(u2r[0], u2r[1], c='r', marker='o')
    
    # plot original points
    ax1.scatter(u1[0], u1[1], c='b', marker='.')
    ax2.scatter(u2[0], u2[1], c='b', marker='.')
    
    plt.savefig('09_reprojection.pdf')
    plt.close()

    
    #! 5.
    # Draw graph of reprojection errors and export as 09_errorsr.pdf.
    errors1 = np.linalg.norm(u1r - u1, axis=0)
    errors2 = np.linalg.norm(u2r - u2, axis=0)
    
    plt.plot(errors1, label='image 1')
    plt.plot(errors2, label='image 2')
    
    plt.xlabel('point index')
    plt.ylabel('reprojection error [px]')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('09_errorsr.pdf')
    # plt.show()
    plt.close()

    #! 6.
    # Draw the 3D point set (using 3D plotting facility) connected by the edges as a wire-frame model, 
    # shown from the top of the tower, from the side, and from some general view. 
    # Export as 09_view1.pdf, 09_view2.pdf, and 09_view3.pdf.
    
    plot_3D_points(X, edges)

    #! 7.
    # Save Fe, E, R, C, P1, P2, X, and u1, u2, point_sel_e as 09b_data.mat.
    sio.savemat('09b_data.mat', {
        'Fe': Fe,
        'E': E,
        'R': R,
        'C': cvec(C),
        'P1': P1,
        'P2': P2,
        'X': X,
        'u1': u1,
        'u2': u2,
        'point_sel_e': ix[indices_best]
    })