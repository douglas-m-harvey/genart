import numpy as np
import numpy.random as rn
import matplotlib.pyplot as plt


def unit_vector(angle):
    return np.array([np.sin(angle), np.cos(angle)])

def corners(point):
    # [top left, top right, bottom left, bottom right]
    tl = [np.floor(point[0]), np.floor(point[1])]
    tr = [np.floor(point[0]), np.ceil(point[1])]
    bl = [np.ceil(point[0]), np.floor(point[1])]
    br = [np.ceil(point[0]), np.ceil(point[1])]
    return np.array([tl, tr, bl, br]).astype(int)

def angle(p1, p2):
    return np.arctan2(p2[0] - p1[0], p2[1] - p1[1])

def vectors(point, corners):
    return np.array([unit_vector(angle(corners[0], point)),
                     unit_vector(angle(corners[1], point)),
                     unit_vector(angle(corners[2], point)),
                     unit_vector(angle(corners[3], point))])

def smoothstep(x):
        if x <= 0: return 0
        # if 0 < x < 1: return 6*x**5 - 15*x**4 + 10*x**3
        if 0 < x < 1: return x
        if 1 <= x: return 1


size = (16, 16)
grid = np.zeros((2, size[0] + 1, size[1] + 1))
grid_choices = rn.choice((0, 1, 2, 3), size)
for index in np.ndindex(size[0] + 1, size[1] + 1):
    grid[:, *index] = unit_vector(rn.uniform(0, 2*np.pi))
grid[:, -1, :] = grid[:, 0, :]
grid[:, :, -1] = grid[:, :, 0]

print(grid_choices)

image_size = (256, 256)
image = np.zeros(image_size)
points = np.array(np.meshgrid(np.linspace(0, size[1], image_size[1]),
                              np.linspace(0, size[0], image_size[0])))
for index in np.ndindex(*image_size):
    point_coords = points[:, *index]
    point_subcoords = np.array([coord%1 for coord in point_coords])
    cell_corners = corners(point_coords)
    cell_vectors = np.array([grid[:, corner[0]%(size[0] + 1), corner[1]%(size[1] + 1)]
                             for corner in cell_corners])
    point_vectors = vectors(point_coords, cell_corners)
    dots = np.array([np.dot(cell_vectors[index], point_vectors[index])
                    for index in range(4)])
    interp_x_t = np.interp(smoothstep(point_subcoords[1]), [0, 1], [dots[0], dots[1]])
    interp_x_b = np.interp(smoothstep(point_subcoords[1]), [0, 1], [dots[2], dots[3]])
    interp_y = np.interp(smoothstep(point_subcoords[0]), [0, 1], [interp_x_t, interp_x_b])
    image[*index] = interp_y

fig, ax = plt.subplots(1, 2, figsize = (8, 4), tight_layout = True)
ax[0].imshow(image)
ax[1].imshow(np.log(np.abs(np.fft.fftshift(np.fft.fft2(image)))))
plt.show()