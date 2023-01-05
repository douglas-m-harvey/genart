import os
import shutil
import argparse
import yaml
import numpy as np
import numpy.random as rn
import matplotlib.pyplot as plt



def angle(p0, p1):
    return np.arctan2(p1[0] - p0[0], p1[1] - p0[1])


def unit_vector(angle):
    return (np.sin(angle), np.cos(angle))


def vector(p0, p1):
    return (p1[0] - p0[0], p1[1] - p0[1])


def corners(point):
    return np.array([[np.floor(point[0]), np.floor(point[1])],
                     [np.floor(point[0]), np.ceil(point[1])],
                     [np.ceil(point[0]), np.floor(point[1])],
                     [np.ceil(point[0]), np.ceil(point[1])]]).astype(int)


def smoothstep(x):
    return 6*x**5 - 15*x**4 + 10*x**3


def value(point, point_dots):
    interp_x_t = np.interp(smoothstep(point[1]%1), [0, 1], point_dots[:2])
    interp_x_b = np.interp(smoothstep(point[1]%1), [0, 1], point_dots[2:])
    return np.interp(smoothstep(point[0]%1), [0, 1], [interp_x_t, interp_x_b])



__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
if not os.path.exists(os.path.join(__location__, "perlin_params.yaml")):
    shutil.copy(os.path.join(__location__, "perlin_params_template.yaml"), os.path.join(__location__, "perlin_params.yaml"))


parser = argparse.ArgumentParser()
parser.add_argument("parameters", nargs = "?", default = "perlin_params", type = str,
                    help = "YAML file with entries: grid_y, grid_x, image_y, image_x",)
args = parser.parse_args()
with open(os.path.join(__location__, args.parameters + ".yaml"), "r") as file:
        params = yaml.safe_load(file)



shape_grid = np.array([params["grid_y"], params["grid_x"]])
shape_image = np.array([params["image_y"], params["image_x"]])


grid_vectors = np.zeros((2, shape_grid[0] + 1, shape_grid[1] + 1))
for index in np.ndindex(grid_vectors.shape[1:]):
    grid_vectors[:, *index] = unit_vector(rn.uniform(0, 2*np.pi))
if params["tile_y"]: grid_vectors[:, -1, :] = grid_vectors[:, 0, :]
if params["tile_x"]: grid_vectors[:, :, -1] = grid_vectors[:, :, 0]

points_y = np.linspace(0, shape_grid[0], shape_image[0])
points_x = np.linspace(0, shape_grid[1], shape_image[1])
points = np.array(np.meshgrid(points_x, points_y))
points[0, ...] %= shape_grid[1]
points[1, ...] %= shape_grid[0]

image = np.zeros(shape_image)
for index in np.ndindex(*shape_image):
    point = points[:, *index][::-1]
    point_dots = [np.dot(grid_vectors[:, *corner], vector(corner, point))
                  for corner in corners(point)]
    image[index] = value(point, point_dots)
    progress = (index[0]*shape_image[1] + index[1])/shape_image.prod()
    print("Progress: {:2.1%}".format(progress), end = "\r")



fig, ax = plt.subplots(1, 1, tight_layout = True)
ax.imshow(image)
ax.axis("off")
plt.show()

save_image = input("\nSave image? [y/n]: ").lower()
if save_image in ["y", "yes"]:
    home_path = os.path.expanduser("~")
    directory_path = input("Save path: " + home_path + "\\")
    full_path = os.path.join(home_path, directory_path)
    if not os.path.exists(full_path): os.mkdir(full_path)
    file_name = input("File name: ")
    plt.imsave(os.path.join(full_path, file_name), image)
elif save_image in ["n", "no", None]:
    pass