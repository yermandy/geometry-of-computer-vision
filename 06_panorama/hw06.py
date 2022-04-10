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
        plt.plot(sorted(errors), label=f'{i + 1}-{i + 2}')

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

    fig, ax = plt.subplots()
    ax.axis('equal')
    
    Hs = [H24, H34, H44, H54, H64]
    
    for H, i in zip(Hs, [2, 3, 4, 5, 6]):
        xs, ys = transform(corners, H)
        ax.plot(xs, ys, label=i)
    
    ax.set_title('Image borders in the coordinate system of image 4')
    ax.set_xlabel('$x_4$ [px]')
    ax.set_ylabel('$y_4$ [px]')
    ax.legend()
    plt.savefig('06_borders.pdf')
    # plt.show()
    plt.close()
    
    #! 7.
    # Construct a projective panoramic image from the images 03, 04, and 05, 
    # in the image plane of the image 04. Save as 06_panorama.png.
    '''
    
    fig, ax = plt.subplots()
    ax.axis('equal')

    Hs = [H34, H44, H54]
    
    min_x = np.inf
    max_x = -np.inf
    min_y = np.inf
    max_y = -np.inf
    for H in Hs:
        xs, ys = transform(corners, H)
        min_x = min(min_x, xs.min())
        max_x = max(max_x, xs.max())
        min_y = min(min_y, ys.min())
        max_y = max(max_y, ys.max())
        
    min_x, min_y = np.floor([min_x, min_y]).astype(int)
    max_x, max_y = np.ceil([max_x, max_y]).astype(int)

    # TODO check this logic
    shift_x = -min_x
    shift_y = -min_y
    
    for H in Hs:
        xs, ys = transform(corners, H)
        plt.plot(xs + shift_x, ys + shift_y, label=i)
    
    images = [image_2, image_3, image_4]
    Hs_inv = [inv(H34), inv(H44), inv(H54)]
    
    # create a blank image for panorama
    panorama = np.zeros((max_y - min_y, max_x - min_x, depth))
    
    # TODO make it more efficient: vectorize
    # fill panorama with pixels from images
    for x in range(min_x, max_x):
        for y in range(min_y, max_y):
            for H_inv, image in zip(Hs_inv, images):
                j, i = p2e(H_inv @ [x, y, 1]).round().astype(int)
                if i < 0 or j < 0:
                    continue
                try:
                    panorama[y + shift_y, x + shift_x] = image[i, j]
                except:
                    pass

    panorama = panorama.astype(np.uint8)
    plt.imshow(panorama)
    # plt.show()
    plt.close()
    
    from PIL import Image
    Image.fromarray(panorama).save('06_panorama.png')
    
    # '''
    
    #! 8.
    # Construct calibration matrix K using the actual image size and the original EXIF. 
    # All the images share the same calibration. Store it in 06_data.mat

    # ExifImageWidth:           2400 px
    # ExifImageHeight:          1800 px
    # FocalPlaneXResolution:    2160000 / 225 inch
    # FocalPlaneYResolution:    1611200 / 168 inch
    # FocalLength:              7.4 mm
        
    # camera calibration matrix from EXIF
    # https://www.studocu.com/en-au/document/australian-national-university/computer-vision/lecture-notes-course-1-camera-calibration/162080
    
    scale = 0.5
    ax = 7.4 * (2160000 / 225) / 25.4 * scale
    ay = 7.4 * (1611200 / 168) / 25.4 * scale
    x0 = 2400 / 2 * scale
    y0 = 1800 / 2 * scale

    K = np.array([
        [ax,  0, x0],
        [ 0, ay, y0],
        [ 0,  0,  1]
    ])
    
    sio.savemat('06_data.mat', {'K': K})