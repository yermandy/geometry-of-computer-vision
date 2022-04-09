from lib import *




if __name__ == '__main__':
    f = sio.loadmat('bridge_corresp.mat')
    u = f['u']
    
    image_6 = plt.imread('bridge_07.jpg').copy()
    image_5 = plt.imread('bridge_06.jpg').copy()
    image_4 = plt.imread('bridge_05.jpg').copy()
    image_3 = plt.imread('bridge_04.jpg').copy() # reference
    image_2 = plt.imread('bridge_03.jpg').copy()
    image_1 = plt.imread('bridge_02.jpg').copy()
    image_0 = plt.imread('bridge_01.jpeg').copy()
    
    images = [image_0, image_1, image_2, image_3, image_4, image_5, image_6]
    images.reverse()
    
    fig, axes = plt.subplots(1, 7, figsize=(20, 3))
    for i, (ax, image) in enumerate(zip(axes, images)):
        ax.imshow(image)
    
    axes[-1].scatter(u[0, 1][0], u[0, 1][1], c='r')
    axes[-2].scatter(u[1, 0][0], u[1, 0][1], c='r')
    axes[-2].scatter(u[1, 2][0], u[1, 2][1], c='g')
    axes[-3].scatter(u[2, 1][0], u[2, 1][1], c='g')
    
    plt.close()
    
    #! 3.
    # Find homographies H_ij for every pair of adjacent images by the same method (optimizing 4 over 10) 
    # as in HW-05; use your u2h_optim function for each homography.

    #! 4.
    # Draw graph of transfer errors, sorted from lower to higher, of all neighboring pairs into a single image. 
    # Use different colour for every image pair, export as 06_errors.pdf.

    plt.figure()
    pairs = [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6]]
    for i, j in pairs:
        H, idx = u2h_optim(u[i, j], u[j, i])
        errors = dist(H, u[i, j], u[j, i])
        plt.plot(range(10), sorted(errors), label=f'{i + 1}-{i + 2}')

    plt.title('transfer errors')
    plt.ylabel('err [px]')
    plt.xlabel('point index (sorted)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('06_errors.pdf')
    # plt.show()
    plt.close()
    
    #! 5.
    # Compute homographies H_i4 for every i ∈ <1,7> that maps the images above 
    # into the reference image 04 (thus H_44 is identity). 
    # Also construct inverse homographies H_4i.

    transform = lambda u, H: p2e(H @ e2p(u))
    
    # reference homography
    H44 = np.eye(3)
    # right side from reference homography
    H34, _ = u2h_optim(u[2, 3], u[3, 2])
    H24, _ = u2h_optim(u[1, 2], transform(u[2, 1], H34))
    H14, _ = u2h_optim(u[0, 1], transform(u[1, 0], H24))
    # left side from reference homography
    H54, _ = u2h_optim(u[4, 3], u[3, 4])
    H64, _ = u2h_optim(u[5, 4], transform(u[4, 5], H54))
    H74, _ = u2h_optim(u[6, 5], transform(u[5, 6], H64))
    
    height, width, depth = image_3.shape
    
    corners = np.array([
        [0, 0], [width, 0], [width, height], [0, height], [0, 0]
    ]).T
    
    #! 6.
    # Plot the image borders of the images 02 to 06 transformed by appropriate 
    # homography to the image plane of the image 04. Export as 06_borders.pdf.

    plt.figure()
    
    Hs = [H24, H34, H44, H54, H64]
    
    for H, i in zip(Hs, [2, 3, 4, 5, 6]):
        transformed_corners = transform(corners, H)
        plt.plot(transformed_corners[0], transformed_corners[1], label=i)
    
    plt.title('Image borders in the coordinate system of image 4')
    plt.xlabel('$x_4$ [px]')
    plt.ylabel('$y_4$ [px]')
    plt.savefig('06_borders.pdf')
    plt.legend()
    # plt.show()
    plt.close()
    
    #! 7.
    # Construct a projective panoramic image from the images 03, 04, and 05, 
    # in the image plane of the image 04. Save as 06_panorama.png.

