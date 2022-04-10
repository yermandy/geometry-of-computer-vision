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
    H34 = u2h_optim(u[2, 3], u[3, 2])[0]
    H24 = u2h_optim(u[1, 2], u[2, 1])[0] @ H34
    H14 = u2h_optim(u[0, 1], u[1, 0])[0] @ H24
    # left side from reference homography
    H54 = u2h_optim(u[4, 3], u[3, 4])[0]
    H64 = u2h_optim(u[5, 4], u[4, 5])[0] @ H54
    H74 = u2h_optim(u[6, 5], u[5, 6])[0] @ H64
    
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
    # '''
    
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
    # calculate shift such that the image origin starts at (0, 0)
    shift_x = -min_x
    shift_y = -min_y
    
    for H in Hs:
        xs, ys = transform(corners, H)
        plt.plot(xs + shift_x, ys + shift_y, label=i)
    
    images = [image_2, image_3, image_4]
    Hs_inv = [inv(H34), inv(H44), inv(H54)]
    
    # create a blank image for panorama
    panorama = np.zeros((max_y - min_y, max_x - min_x, depth), dtype=np.uint8)
    
    grid = np.meshgrid(range(min_x, max_x), range(min_y, max_y))
    coords = np.concatenate(grid).reshape(2, -1)
    coords_shifted = coords + cvec([shift_x, shift_y])
    
    for H_inv, image in zip(Hs_inv, images):
        xs, ys = p2e(H_inv @ e2p(coords)).round().astype(int)
        
        mask = (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height)
        
        xs = xs[mask]
        ys = ys[mask]
        cs = coords_shifted[:, mask]
        
        panorama[cs[1], cs[0]] = image[ys, xs]
    
    plt.imshow(panorama)
    plt.show()
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
    
    #! 9.
    # Establish a transformation between each image plane and the coordinate system of the cylinder, described below.
    #! 10.
    # Plot the image borders of all the images mapped onto the cylinder (06_borders_c.pdf).

    fig, ax = plt.subplots()
    ax.axis('equal')
    
    N = 20
    C1 = np.array([np.linspace(0, width, N), [0] * N])
    C2 = np.array([[width] * N, np.linspace(0, height, N)])
    C3 = np.array([np.linspace(width, 0, N), [height] * N])
    C4 = np.array([[0] * N, np.linspace(height, 0, N)])
    coords_alpha = np.concatenate([C1, C2, C3, C4, C1], axis=1)
    
    Hs = [H14, H24, H34, H44, H54, H64, H74]
    K_inv = inv(K)
    
    x_min = np.inf
    y_min = np.inf
    x_max = -np.inf
    y_max = -np.inf
    
    for i, H in enumerate(Hs):
        # alpha to beta
        coords_beta = H @ e2p(coords_alpha)
        
        # beta to gamma
        coords_gamma = K_inv @ coords_beta
        
        xs, ys, zs = coords_gamma
        
        a = np.arctan2(xs, zs)
        y = ys / np.sqrt(xs ** 2 + zs ** 2)
        
        coords = K[0, 0] * np.array([a, y])
        plt.plot(coords[0], coords[1], label=i + 1)
        
        x_min = min(coords[0].min(), x_min)
        y_min = min(coords[1].min(), y_min)
        x_max = max(coords[0].max(), x_max)
        y_max = max(coords[1].max(), y_max)
        
    ax.set_title('Image borders in the cylindrical plane')
    ax.set_xlabel('$x_c$ [px]')
    ax.set_ylabel('$y_c$ [px]')
    plt.legend()
    plt.savefig('06_borders_c.pdf')
    # plt.show()
    plt.close()
    
    shift_x = -x_min
    shift_y = -y_min
    
    x_min, y_min = np.floor([x_min, y_min]).astype(int)
    x_max, y_max = np.ceil([x_max, y_max]).astype(int)
    
    panorama = np.zeros((y_max - y_min, x_max - x_min, depth), dtype=np.uint8)
    
    images = [image_0, image_1, image_2, image_3, image_4, image_5, image_6]
    
    # TODO inverse mapping as in previous task
    for H, image in zip(Hs, images):
        grid = np.meshgrid(range(width), range(height))
        coords_alpha = np.concatenate(grid).reshape(2, -1)
        
        # alpha to beta
        coords_beta = H @ e2p(coords_alpha)
        
        # beta to gamma
        coords_gamma = K_inv @ coords_beta
        
        xs, ys, zs = coords_gamma
        
        a = np.arctan2(xs, zs)
        y = ys / np.sqrt(xs ** 2 + zs ** 2)
        
        coords = K[0, 0] * np.array([a, y])
        coords += cvec([shift_x, shift_y])
        coords = coords.round().astype(int)
        
        panorama[coords[1], coords[0]] = image[coords_alpha[1], coords_alpha[0]]
        
    plt.imshow(panorama)
    plt.show()
    plt.close()
    
    from PIL import Image
    Image.fromarray(panorama).save('06_panorama_c.png')