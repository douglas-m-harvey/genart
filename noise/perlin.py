import os
import shutil
import argparse
import yaml
import numpy as np
import numpy.random as rn
import matplotlib.pyplot as plt



def unit_vector(angle):
    return np.array([np.sin(angle), np.cos(angle)])

def smoothstep(x):
    return 6*x**5 - 15*x**4 + 10*x**3



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


# Define a grid of random unit vectors, with equal edge vectors so image can be tiled.
vectors = unit_vector(rn.uniform(0, 2*np.pi, shape_grid + 1))
vectors[:, -1, :] = vectors[:, 0, :]
vectors[:, :, -1] = vectors[:, :, 0]

# Define a grid of sample points.
points = np.array(np.meshgrid(np.linspace(0, shape_grid[1], shape_image[1]),
                              np.linspace(0, shape_grid[0], shape_image[0])))

# For each sample point, find the top, bottom, left and right edges of its cell.
# Format: [shape_image]
t = np.floor(points[0]).astype(int)
b = np.ceil(points[0]).astype(int)
l = np.floor(points[1]).astype(int)
r = np.ceil(points[1]).astype(int)

# For each sample point, find the gradient vectors of its cell.
# Format: [2, *shape_image]
gv_tl = np.array([vectors[0][l, t], vectors[1][l, t]])
gv_tr = np.array([vectors[0][r, t], vectors[1][r, t]])
gv_bl = np.array([vectors[0][l, b], vectors[1][l, b]])
gv_br = np.array([vectors[0][r, b], vectors[1][r, b]])

# Define an array containing gradient vectors for every point.
# Format: [4, 2, *shape_image]
gv = np.array([gv_tl, gv_tr, gv_bl, gv_br])

# For each sample point, calculate the distances to the corners of its cell.
# Format: [shape_image]
d_t = t - points[0]
d_b = b - points[0]
d_l = l - points[1]
d_r = r - points[1]

# Define the distance vectors of each point.
# Format: [2, *shape_image]
dv_tl = np.array([d_t, d_l])
dv_tr = np.array([d_t, d_r])
dv_bl = np.array([d_b, d_l])
dv_br = np.array([d_b, d_r])

# Define an array containing distance vectors for every point.
# Format: [4, 2, *shape_image]
dv = np.array([dv_tl, dv_tr, dv_bl, dv_br])

# Calculate the dot product between each point's distance vectors and its cell's gradient vectors.
# Format: [4, *shape_image]
dots = dv[:, 0, :, :]*gv[:, 0, :, :] + dv[:, 1, :, :]*gv[:, 1, :, :]

# Interpolate between each point's dot products to define the image.
# Format[shape_image]
interp_x_t = dots[0, ...] + smoothstep(points[1, ...]%1)*np.subtract(dots[1, ...], dots[0, ...])
interp_x_b = dots[2, ...] + smoothstep(points[1, ...]%1)*np.subtract(dots[3, ...], dots[2, ...])
image = interp_x_t + smoothstep(points[0, ...][::-1]%1)*np.subtract(interp_x_b, interp_x_t)



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