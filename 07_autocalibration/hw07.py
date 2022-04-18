from re import A
from lib import *
import scipy
import scipy.linalg

def plot_vanishing_points(corners, ax, color='b'):
    x1, x2, x3, x4 = e2p(corners).T

    v1 = x2vp(x1, x2, x3, x4)
    v2 = x2vp(x2, x3, x4, x1)
    
    # lines to the first vanishing point
    ax.plot([x1[0], x2[0], v1[0]], [x1[1], x2[1], v1[1]], color=color)
    ax.plot([x3[0], x4[0], v1[0]], [x3[1], x4[1], v1[1]], color=color)
    
    # lines to the second vanishing point
    ax.plot([x2[0], x3[0], v2[0]], [x2[1], x3[1], v2[1]], color=color)
    ax.plot([x4[0], x1[0], v2[0]], [x4[1], x1[1], v2[1]], color=color)
    
    # two vanishing points
    ax.scatter([v1[0], v2[0]], [v1[1], v2[1]], color=color, marker='x')
    
    return v1, v2

    
def plot_horizon(vps, ax):
    i = np.argmin(vps[0])
    j = np.argmax(vps[0])
    
    ax.plot([vps[0, i], vps[0, j]], [vps[1, i], vps[1, j]], color='g')
    

def savefig_zoom(name, width, height):
    plt.xlim(0, width)
    plt.ylim(0, height)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(name)
    
    
def K_from_three_vanishing_points(vp1, vp2):
    # Solves eq. (11.41), p. 101
    A = []
    b = []
    
    # four pairs from two images
    pair1 = (vp1[:, 0], vp1[:, 1])
    pair2 = (vp1[:, 2], vp1[:, 3])
    pair3 = (vp2[:, 0], vp2[:, 1])
    pair4 = (vp2[:, 2], vp2[:, 3])
    
    # four possible triples -> four possible K matrices
    triple = (pair1, pair2, pair3)
    # triple = (pair1, pair2, pair4)
    # triple = (pair1, pair3, pair4)
    # triple = (pair2, pair3, pair4)
    for (v11, v12), (v21, v22) in triple:
        A.append([v11 + v21, v12 + v22, 1])
        b.append(-(v21 * v11 + v22 * v12))
        
    o13, o23, o33 = np.linalg.solve(A, b)
    
    k13 = -o13
    k23 = -o23
    k11 = math.sqrt(o33 - k13 ** 2 - k23 ** 2)
    
    K = np.array([
        [k11,   0, k13],
        [  0, k11, k23],
        [  0,   0,   1]
    ])
    
    return K
    
    
def plot_cube(P, ax):
    cube = np.array([
        [0, 0, 0], 
        [0, 0, 1], 
        [0, 1, 0], 
        [0, 1, 1],
        [1, 0, 0],
        [1, 0, 1],
        [1, 1, 0],
        [1, 1, 1]
    ])
    
    for point_1 in cube:
        x1, y1 = p2e(P @ e2p(cvec(point_1)))
        for point_2 in cube:
            if norm(point_1 - point_2) == 1:
                x2, y2 = p2e(P @ e2p(cvec(point_2)))
                ax.plot([x1, x2], [y1, y2], 'bo-')    
    
    
if __name__ == '__main__':

    # Image 1 black square corners (clock-wise) (image upper left corner is [0, 0] ):
    C = np.array([
        [502.3,   565.1,   787.6,   753.4],
        [485.0,   341.2,   362.5,   516.0]
    ])

    # Image 2 black square corners (clock-wise) (image upper left corner is [0, 0] ):
    C2 = np.array([
        [652.1,   458.0,   565.2,   745.7],
        [532.9,   456.3,   340.4,   403.7]
    ])

    # Image 1 outer rectangle corners (clock-wise) (image upper left corner is [1,1] ):
    Co = np.array([
        [160.4,   550.1,  1045.4,   749.4],
        [542.9,   213.4,   359.4,   820.3]
    ])

    # Image 2 outer rectangle corners (clock-wise) (image upper left corner is [1,1] ):
    Co2 = np.array([
        [636.1,   229.4,   626.0,  1038.1],
        [791.7,   426.2,   223.0,   475.3]
    ])

    #! 1. Extract vanishing points
    #! 1.1
    # Download two images of 'pokemons' from the upload system (InputData).
    
    #! 1.2
    # Download the coordinates of the corners of the poster and of the black square, 
    # u1 in the first image and u2 in the second. The corners are in the clock-wise order 
    # and correspond between the images (this is not used for calibration but later).
    
    #! 1.3
    # Construct vanishing points from the squares, store them as vp1 and vp2 
    # (four points each) for the first and the second image, respectively.
    
    #! 1.4
    # Show both images, draw the vanishing points into both images and connect (by a line) 
    # each vanishing point with all corners of appropriate rectangle. 
    # Also connect the two most distant vanishing points by a line in each of the images. 
    # Export as 07_vp1.pdf and 07_vp2.pdf. Then show each figure zoomed such that the image 
    # is clearly visible and export as 07_vp1_zoom.pdf, 07_vp2_zoom.pdf.

    fig, ax = plt.subplots(1, 1)
    image = plt.imread('pokemon_09.jpeg')
    height, width, depth = image.shape
    ax.imshow(image)
    
    v1, v2 = plot_vanishing_points(C, ax, color='b')
    v3, v4 = plot_vanishing_points(Co, ax, color='r')
    
    vp1 = np.c_[v1, v2, v3, v4]
    plot_horizon(vp1, ax)
    plt.tight_layout()
    plt.savefig('07_vp1.pdf')
    savefig_zoom('07_vp1_zoom.pdf', width, height)
    plt.close()
    
    fig, ax = plt.subplots(1, 1)
    image = plt.imread('pokemon_34.jpeg')
    height, width, depth = image.shape
    ax.imshow(image)
    
    v5, v6 = plot_vanishing_points(C2, ax, color='b')
    v7, v8 = plot_vanishing_points(Co2, ax, color='r')
    
    vp2 = np.c_[v5, v6, v7, v8]
    plot_horizon(vp2, ax)
    plt.savefig('07_vp2.pdf')
    savefig_zoom('07_vp2_zoom.pdf', width, height)
    plt.close()
    
    #! 2. Calibration
    #! 2.1 
    # Compute the camera calibration matrix K from three vanishing points. 
    # From the four available v.p. pairs (two in each image), select three pairs.
    
    #! 2.2 
    # Compute the angle (should be acute) in the scene between the square and the rectangle. 
    # Use the mean value of four computed angles from four pairs of vanishing points.

    K = K_from_three_vanishing_points(vp1, vp2)

    u1 = np.c_[C, Co]
    u2 = np.c_[C2, Co2]
    
    v1 = e2p(cvec(v1))
    v2 = e2p(cvec(v2))
    v3 = e2p(cvec(v3))
    v4 = e2p(cvec(v4))
    v5 = e2p(cvec(v5))
    v6 = e2p(cvec(v6))
    v7 = e2p(cvec(v7))
    v8 = e2p(cvec(v8))
    
    K_inv = inv(K)
    # ω is called the image of the absolute conic, p. 95.
    omega = K_inv.T @ K_inv
    
    # Eq. (11.1)
    cos_x_y = lambda x, y: ((x.T @ omega @ y) / (math.sqrt(x.T @ omega @ x) * math.sqrt(y.T @ omega @ y)))[0, 0]
    angle_x_y = lambda x, y: math.acos(cos_x_y(x, y))
    
    angle_v1_v3 = angle_x_y(v1, v3)
    angle_v2_v4 = angle_x_y(v2, v4)
    angle_v5_v7 = angle_x_y(v5, v7)
    angle_v6_v8 = angle_x_y(v6, v8)
    
    angle = (angle_v1_v3 + angle_v2_v4 + angle_v5_v7 + angle_v6_v8) / 4
        
    sio.savemat('07_data.mat', {
        'u1': u1,
        'u2': u2,
        'vp1': vp1,
        'vp2': vp2,
        'K': K,
        'angle': angle,
        'C1': np.zeros((3, 1)),
        'C2': np.zeros((3, 1)),
        'R1': np.eye(3),
        'R2': np.eye(3),
    })
    
    
    #! 3. Virtual object
    #! 3.1
    # Use the K to compute the pose of calibrated camera w.r.t. the black square using P3P. 
    # Compute camera centers C1, C2, and rotations R1, R2 (both images). 
    # Chose one corner of the square as origin and consider the square sides having the unit length.
    
    #! 3.2
    # Create a virtual object: 'place' a cube into the two images. 
    # The black square is the bottom face of the cube, which sits on the poster. 
    # Show the wire-frame cube in each of the images, export as 07_box_wire1.pdf, 07_box_wire2.pdf.
    
    u = C[:, [0, 3, 1]].copy()
    X = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
    
    R1, C1 = uXK2RC(u, X, K)    
    P1 = K @ R1 @ np.c_[np.eye(3), -C1]
    
    fig, ax = plt.subplots()
    image = plt.imread('pokemon_09.jpeg')
    ax.imshow(image)
    plot_cube(P1, ax)
    plt.savefig('07_box_wire1.pdf')
    # plt.show()
    plt.close()
    
    
    u = C2[:, [0, 3, 1]].copy()
    X = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
    
    R2, C2 = uXK2RC(u, X, K, 1)
    P2 = K @ R2 @ np.c_[np.eye(3), -C2]
    
    fig, ax = plt.subplots()
    image = plt.imread('pokemon_34.jpeg')
    ax.imshow(image)
    plot_cube(P2, ax)
    plt.savefig('07_box_wire2.pdf')
    # plt.show()
    plt.close()