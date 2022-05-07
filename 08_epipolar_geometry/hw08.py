from lib import *



def u2F(u1, u2):
    """ Computes the fundamental matrix using the seven-point algorithm 
    from 7 euclidean correspondences u1, u2, measured in two images.

    Parameters
    ----------
    u1 : np.array (2, 7)
        coordinates of the seven correspondences in the first image
    u2 : np.array (2, 7)
        coordinates of the seven correspondences in the second image
    """
    
    pass


if __name__ == '__main__':
    u = sio.loadmat('daliborka_01_23-uu.mat')
    
    edges = u['edges'] - 1
    u1 = u['u01']
    u2 = u['u23']
    ix = u['ix'] - 1
    
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
    plt.show()
    
    #! 1.
    # Find the fundamental matrix F relating the images above: 
    # generate all 7-tuples from the selected set of 12 correspondences, 
    # estimate F for each of them and chose the one, 
    # that minimizes maximal epipolar error over all matches.

    #! 2.
    # Draw the 12 corresponding points in different colour in the two images. 
    # Using the best F, compute the corresponding epipolar lines and draw them into the images in corresponding colours 
    # (a line segment given by the intersection of the image area and a line must be computed). 
    # Export as 08_eg.pdf.
    
    #! 3.
    # Draw graphs of epipolar errors d1_i and d2_i for all points 
    # (point index on horizontal axis, the error on vertical axis). 
    # Draw both graphs into single figure (different colours) and export as 08_errors.pdf.

    #! 4.
    # Save all the data into 08_data.mat: the input data u1, u2, ix, 
    # the indices of the 7 points used for computing the optimal F as point_sel and the matrix F.
