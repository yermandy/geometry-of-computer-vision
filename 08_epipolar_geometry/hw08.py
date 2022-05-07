from re import L
from lib import *



def u2F(x1, x2):
    """ Computes the fundamental matrix using the seven-point algorithm 
    from 7 euclidean correspondences x1, x2, measured in two images.

    Parameters
    ----------
    x1 : np.array (2, 7)
        coordinates of the seven correspondences in the first image
    x2 : np.array (2, 7)
        coordinates of the seven correspondences in the second image
        
    Returns
    -------
    Fs : list of (3, 3) np.array
        fundamental matrices
    """
    u1, v1 = x1
    u2, v2 = x2
    
    A = np.c_[u2 * u1, u2 * v1, u2, v2 * u1, v2 * v1, v2, u1, v1, np.ones(7)]
    
    # Solve eq. (12.33)
    G1, G2 = scipy.linalg.null_space(A).T
    G1 = G1.reshape(3, 3)
    G2 = G2.reshape(3, 3)
    
    # Solve eq. (12.36)
    polynomial = u2F_polynom(G1, G2)
    # a_3 * α^3 + a_2 * α^2 + a_1 * α + a_0 = 0
    alphas = np.roots(polynomial)
    
    Fs = []
    
    for alpha in alphas:
        if np.iscomplex(alpha):
            continue
        alpha = np.real(alpha)
        
        # Eq. (12.34)
        G = G1 + alpha * G2
                
        if np.linalg.matrix_rank(G) != 2:
            continue    
        
        # Notice that we assumed that G was constructed with a non-zero coefficient at G1. 
        # We therefore also need to check G = G2 for a solution.
        if np.allclose(G, G2):
            continue

        Fs.append(G)
        
    return Fs


def epipolar_errors(u, l):
    """ Computes the epipolar error of points u on lines l.

    Parameters
    ----------
    u : np.array (3, n)
        coordinates of the points
    l : np.array (3, n)
        lines

    Returns
    -------
    np.array (n,)
        epipolar errors
    """
    errors = []
    for u_i, l_i in zip(u.T, l.T):
        error = abs(u_i @ l_i) / np.sqrt(l_i[0] ** 2 + l_i[1] ** 2)
        errors.append(error)
    return np.array(errors)


def u2F_optimal(u1_all, u2_all, ix):
    u1_selected = u1_all[:, ix]
    u2_selected = u2_all[:, ix]
    
    u1p_all = e2p(u1_all)
    u2p_all = e2p(u2_all)
    
    min_max_epipolar_error = np.inf
    indices_best = None
    F_best = None
    
    for indices in itertools.combinations(range(len(u1_selected[0])), 7):
        indices = np.array(indices)
        u1 = u1_selected[:, indices]
        u2 = u2_selected[:, indices]
        Fs = u2F(u1, u2)
        
        for F in Fs:
            l2 = F @ u1p_all
            l1 = F.T @ u2p_all
            
            errors1 = epipolar_errors(u1p_all, l1)
            errors2 = epipolar_errors(u2p_all, l2)
            
            errors = errors1 + errors2
            
            max_epipolar_error = np.max(errors)
            
            if max_epipolar_error < min_max_epipolar_error:
                min_max_epipolar_error = max_epipolar_error
                indices_best = indices
                F_best = F
        
    return F_best, indices_best
    
    
def line_points(a=0, b=0, c=0, xs=[0, 1]):
    if (a == 0 and b == 0):
        raise Exception("both a and b cannot be zero")
    return [(-c / a, x) if b == 0 else (x, (-c - a * x) / b) for x in xs]


def show_epipolar_lines(u, F, ax1, ax2):
    for u1i, u2i in u.T:
        sc = ax2.scatter(u1i, u2i, s=10)
        l = F @ [u1i, u2i, 1]
        ax1.axline(*line_points(*l), color=sc.get_facecolors()[-1])
    

if __name__ == '__main__':
    file = sio.loadmat('daliborka_01_23-uu.mat')
    
    edges = file['edges'] - 1
    u1 = file['u01']
    u2 = file['u23']
    ix = file['ix'][0] - 1
    
    image1 = plt.imread('daliborka_01.jpg')
    image2 = plt.imread('daliborka_23.jpg')
    
    fig, axes = plt.subplots(1, 2)
    axes[0].imshow(image1)
    axes[0].scatter(u1[0], u1[1], s=10)
    axes[0].scatter(u1[0, ix], u1[1, ix], s=10)
    
    axes[1].imshow(image2)
    axes[1].scatter(u2[0], u2[1], s=10)
    axes[1].scatter(u2[0, ix], u2[1, ix], s=10)
    
    plt.tight_layout()
    # plt.show()
    plt.close()
    
    #! 1.
    # Find the fundamental matrix F relating the images above: 
    # generate all 7-tuples from the selected set of 12 correspondences, 
    # estimate F for each of them and chose the one, 
    # that minimizes maximal epipolar error over all matches.
    F, indices_best = u2F_optimal(u1, u2, ix)
    

    #! 2.
    # Draw the 12 corresponding points in different colour in the two images. 
    # Using the best F, compute the corresponding epipolar lines and draw them into the images in corresponding colours 
    # (a line segment given by the intersection of the image area and a line must be computed). 
    # Export as 08_eg.pdf.
    fig, axes = plt.subplots(1, 2)
    ax1, ax2 = axes
    
    # show images
    ax1.imshow(image1)
    ax2.imshow(image2)
    
    # show epipolar lines
    show_epipolar_lines(u1[:, ix], F, ax2, ax1)
    show_epipolar_lines(u2[:, ix], F.T, ax1, ax2)
    
    # set limits
    height, width, _ = image1.shape
    for ax in axes:
        ax.set_ylim(0, height)
        ax.set_xlim(0, width)
        ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig('08_eg.pdf')
    # plt.show()
    plt.close()
    
    
    #! 3.
    # Draw graphs of epipolar errors d1_i and d2_i for all points 
    # (point index on horizontal axis, the error on vertical axis). 
    # Draw both graphs into single figure (different colours) and export as 08_errors.pdf.
    
    u1p = e2p(u1)
    u2p = e2p(u2)
    
    l2 = F @ u1p
    l1 = F.T @ u2p
    
    errors1 = epipolar_errors(u1p, l1)
    errors2 = epipolar_errors(u2p, l2)
    
    plt.plot(errors1, label='image 1')
    plt.plot(errors2, label='image 2')
    
    plt.xlabel('point index')
    plt.ylabel('epipolar error [px]')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('08_errors.pdf')
    # plt.show()
    plt.close()

    #! 4.
    # Save all the data into 08_data.mat: the input data u1, u2, ix, 
    # the indices of the 7 points used for computing the optimal F as point_sel and the matrix F.    
    sio.savemat('08_data.mat', {'u1': u1, 'u2': u2, 'ix': ix, 'point_sel': ix[indices_best], 'F': F})
