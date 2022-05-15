from lib import *


def plot_errors(u1, u2, F, name):
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
    plt.savefig(name)
    # plt.show()
    plt.close()

    
def plot_epipolar_lines(image1, image2, u1, u2, ix, F, name):
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
    plt.savefig(name)
    # plt.show()
    plt.close()


if __name__ == '__main__':
    #! Part A: Essential matrix
    
    K = sio.loadmat('K.mat')['K']
    K_inv = inv(K)
    
    file = sio.loadmat('daliborka_01_23-uu.mat')
    edges = file['edges'] - 1
    u1 = file['u01']
    u2 = file['u23']
    ix = file['ix'][0] - 1
    
    image1 = plt.imread('daliborka_01.jpg')
    image2 = plt.imread('daliborka_23.jpg')
    
    # Find two essential matrices. A possibly bad Ex and the best E
    
    #! 1.
    # Compute essential matrix Ex using your best fundamental matrix F 
    # estimated in HW-08 and internal calibration from HW-04. 
    # Compute also the fundamental matrix Fx consistent with K from Ex and K
    F, _ = u2F_optimal(u1, u2, ix)
    Ex = K.T @ F @ K
    U, D, V_T = np.linalg.svd(Ex)
    Ex = U @ np.diag([1, 1, 0]) @ V_T
    Fx = K_inv.T @ Ex @ K_inv

    #! 2.
    # Draw the 12 corresponding points IX (from HW-08) in different colour in the two images. 
    # Using Fx, compute the corresponding epipolar lines and draw them into the images in corresponding colours. 
    # Export as 09_egx.pdf.
    plot_epipolar_lines(image1, image2, u1, u2, ix, Fx, '09_egx.pdf')

    #! 3.
    # Draw graphs of epipolar errors d1_i and d2_i w.r.t Fx for all points. 
    # Draw both graphs into single figure (different colours) and export as 09_errorsx.pdf.
    plot_errors(u1, u2, Fx, '09_errorsx.pdf')

    #! 4.
    # Find essential matrix E by minimizing the maximum epipolar error 
    # of the respective fundamental matrix Fe consistent with K using the same correspondences
    Fe, E, indices_best = uK2FeE_optimal(u1, u2, ix, K)

    #! 5.
    # Draw the 12 corresponding points in different colour in the two images. 
    # Using Fe, compute the corresponding epipolar lines and draw them into the images in corresponding colours. 
    # Export as 09_eg.pdf.
    plot_epipolar_lines(image1, image2, u1, u2, ix, Fe, '09_eg.pdf')
    
    #! 6.
    # Draw graphs of epipolar errors d1_i and d2_i w.r.t Fe for all points. 
    # Draw both graphs into single figure (different colours) and export as 09_errors.pdf.
    plot_errors(u1, u2, Fe, '09_errors.pdf')

    #! 7.
    # Save F, Ex, Fx, E, Fe and u1, u2, point_sel_e (indices of seven points used for computing Fe) as 09a_data.mat.
    sio.savemat('09a_data.mat', {
        'F': F,
        'Ex': Ex,
        'Fx': Fx,
        'E': E,
        'Fe': Fe,
        'u1': u1,
        'u2': u2,
        'point_sel_e': ix[indices_best]
    })