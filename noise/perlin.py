import os
import shutil
import argparse
import yaml
import numpy as np
import cupy as cp
import matplotlib.pyplot as plt



def unit_vector(angle):
    return np.array([np.sin(angle), np.cos(angle)])

def smoothstep(x):
    return 6*x**5 - 15*x**4 + 10*x**3

def gen_grid(shape_grid, rng):
    # Define a grid of random unit vectors.
    shape_grid = (shape_grid[0] + 1, shape_grid[1] + 1)
    grid = unit_vector(rng.random(shape_grid)*2*np.pi)
    return grid

def tile_grid(grid, tile):
    grid_tiled = grid.copy()
    if tile[0]: grid_tiled[:, -1, :] = grid_tiled[:, 0, :]
    if tile[1]: grid_tiled[:, :, -1] = grid_tiled[:, :, 0]
    return grid_tiled

def gen_points(shape_grid, shape_image):
    points = np.array(np.meshgrid(np.linspace(0, shape_grid[1], shape_image[1]),
                                  np.linspace(0, shape_grid[0], shape_image[0])))
    return points

def warp_points(points, warp):
    points_warped = points.copy() + warp
    # Could use cp.apply_along_axis here 
    points_warped[0] %= points[0, 0, -1]
    points_warped[1] %= points[1, -1, -1]
    return points_warped

def gen_image(grid, points):
    # Define points and grids as numpy arrays then only load in to GPU when this function is called
    grid_gpu = cp.array(grid)
    points_gpu = cp.array(points)
    
    # For each sample point, find the top, bottom, left and right edges of its cell.
    # Format: [shape_image]
    t = cp.floor(points_gpu[0]).astype(int)
    b = cp.ceil(points_gpu[0]).astype(int)
    l = cp.floor(points_gpu[1]).astype(int)
    r = cp.ceil(points_gpu[1]).astype(int)
    
    # For each sample point, find the gradient vectors of its cell.
    # Format: [2, *shape_image]
    gv_tl = cp.array([grid_gpu[0][l, t], grid_gpu[1][l, t]])
    gv_tr = cp.array([grid_gpu[0][r, t], grid_gpu[1][r, t]])
    gv_bl = cp.array([grid_gpu[0][l, b], grid_gpu[1][l, b]])
    gv_br = cp.array([grid_gpu[0][r, b], grid_gpu[1][r, b]])
    
    del grid_gpu

    # Define an array containing gradient vectors for every point.
    # Format: [4, 2, *shape_image]
    gv = cp.array([gv_tl, gv_tr, gv_bl, gv_br])
    
    del gv_tl
    del gv_tr
    del gv_bl
    del gv_br
    
    # For each sample point, calculate the distances to the corners of its cell.
    # Format: [shape_image]
    d_t = t - points_gpu[0]
    d_b = b - points_gpu[0]
    d_l = l - points_gpu[1]
    d_r = r - points_gpu[1]
    
    # Define the distance vectors of each point.
    # Format: [2, *shape_image]
    dv_tl = cp.array([d_t, d_l])
    dv_tr = cp.array([d_t, d_r])
    dv_bl = cp.array([d_b, d_l])
    dv_br = cp.array([d_b, d_r])
    
    del d_t
    del d_b
    del d_l
    del d_r

    # Define an array containing distance vectors for every point.
    # Format: [4, 2, *shape_image]
    dv = cp.array([dv_tl, dv_tr, dv_bl, dv_br])
    
    del dv_tl
    del dv_tr
    del dv_bl
    del dv_br
    
    # Calculate the dot product between each point's distance vectors and its cell's gradient vectors.
    # Format: [4, *shape_image]
    dots = dv[:, 0, :, :]*gv[:, 0, :, :] + dv[:, 1, :, :]*gv[:, 1, :, :]
    
    del dv
    del gv

    # Interpolate between each point's dot products to define the image.
    # Format[shape_image]
    interp_x_t = dots[0, ...] + smoothstep(points_gpu[1, ...]%1)*cp.subtract(dots[1, ...], dots[0, ...])
    interp_x_b = dots[2, ...] + smoothstep(points_gpu[1, ...]%1)*cp.subtract(dots[3, ...], dots[2, ...])
    
    del dots
    
    interp_y = interp_x_t + smoothstep(points_gpu[0, ...]%1)*cp.subtract(interp_x_b, interp_x_t)
    
    del points_gpu
    del interp_x_t
    del interp_x_b
    
    image = cp.asnumpy(interp_y)
    
    del interp_y
    
    cp.get_default_memory_pool().free_all_blocks()
    
    return image


class perlin:
    
    def __init__(self, shape_grid = None, shape_image = None, seed = None, tile = (True, True), warp = None):
        
        self.seed = seed
        self.tile = tile
        self.shape_grid = shape_grid
        self.shape_image = shape_image
        self.warp = warp
        self.update_grid()
        self.update_points()
    
    @property
    def seed(self):
        return self._seed
    
    @seed.setter
    def seed(self, value):
        if value is not None:
            if type(value) is not int:
                raise ValueError("seed must be a non-negative integer or None.")
            elif value < 0:
                raise ValueError("seed must be non-negative.")
            if hasattr(self, "seed"):
                if value == self.seed:
                    return
        self._seed = value
        self.rng = np.random.default_rng(self._seed)
        if hasattr(self, "grid"):
            delattr(self, "_grid_untiled")
            self.update_grid()
        
    @property
    def tile(self):
        return self._tile
    
    @tile.setter
    def tile(self, value):
        if not hasattr(value, "__iter__"):
            raise ValueError("tile must be an iterable containing 2 booleans.")
        elif len(np.shape(value)) != 1:
            raise ValueError("tile must be 1-dimensional.")
        elif np.shape(value)[0] != 2:
            raise ValueError("tile must have 2 elements.")
        elif value[0] not in [0, 1] or value[1] not in [0, 1]:
            raise ValueError("elements of tile must be booleans.")
        if hasattr(self, "tile"):
            if value[0] == self.tile[0] and value[1] == self.tile[1]:
                return
        self._tile = value
        if hasattr(self, "grid"):
            self.update_grid()
            
    @property
    def shape_grid(self):
        return self._shape_grid
    
    @shape_grid.setter
    def shape_grid(self, value):
        if value is not None:
            if not hasattr(value, "__iter__"):
                raise ValueError("shape_grid must be an iterable containing 2 integers greater than or equal to 1 or None.")
            elif len(np.shape(value)) != 1:
                raise ValueError("shape_grid must be 1-dimensional.")
            elif np.shape(value)[0] != 2:
                raise ValueError("shape_grid must have 2 elements.")
            elif not np.issubdtype(type(value[0]), np.integer) or not np.issubdtype(type(value[1]), np.integer):
                raise ValueError("elements of shape_grid must be integers.")
            elif value[0] < 1 or value[1] < 1:
                raise ValueError("elements of shape_grid must be greater than or equal to 1.")
            if hasattr(self, "shape_grid"):
                if self.shape_grid is not None:
                    if value[0] == self.shape_grid[0] and value[1] == self.shape_grid[1]:
                        return
                    elif value[0] != self.shape_grid[0] or value[1] != self.shape_grid[1]:
                        delattr(self, "_grid_untiled")
                        delattr(self, "_points_unwarped")
                if self.shape_grid is None:
                    if hasattr(self, "_grid_untiled"):
                        delattr(self, "_grid_untiled")
        self._shape_grid = value
        if hasattr(self, "grid"):
            self.update_grid()
    
    @property
    def shape_image(self):
        return self._shape_image
    
    @shape_image.setter
    def shape_image(self, value):
        if value is not None:
            if not hasattr(value, "__iter__"):
                raise ValueError("shape_image must be an iterable containing 2 integers greater than or equal to 1 or None.")
            elif len(np.shape(value)) != 1:
                raise ValueError("shape_image must be 1-dimensional.")
            elif np.shape(value)[0] != 2:
                raise ValueError("shape_image must have 2 elements.")
            elif not np.issubdtype(type(value[0]), np.integer) or not np.issubdtype(type(value[1]), np.integer):
                raise ValueError("elements of shape_image must be integers.")
            elif value[0] < 1 or value[1] < 1:
                raise ValueError("elements of shape_image must be greater than or equal to 1.")
            if hasattr(self, "shape_image"):
                if self.shape_image is not None:
                    if value[0] == self.shape_image[0] and value[1] == self.shape_image[1]:
                        return
                    elif value[0] != self.shape_image[0] or value[1] != self.shape_image[1]:
                        delattr(self, "_points_unwarped")
                    if self.shape_image is None:
                        if hasattr(self, "_points_unwarped"):
                            delattr(self, "_points_unwarped")
        self._shape_image = value
        if hasattr(self, "points"):
            self.update_points()
    
    @property
    def warp(self):
        return self._warp
    
    @warp.setter
    def warp(self, value):
        if value is not None:
            if not hasattr(value, "__iter__"):
                raise ValueError("warp must be an array of shape (2, *shape_image).")
            elif len(np.shape(value)) != 3:
                raise ValueError("warp must be 3-dimensional.")
            elif np.shape(value) != (2, *self.shape_image):
                raise ValueError("warp must have shape (2, *shape_image).")
        self._warp = value
        if hasattr(self, "points"):
            self.update_points()
    
    def update_grid(self):
        if self.shape_grid is not None:
            if not hasattr(self, "_grid_untiled"):
                self._grid_untiled = gen_grid(self.shape_grid, self.rng)
            if self.tile[0] or self.tile[1]:
                self.grid = tile_grid(self._grid_untiled, self.tile)
            elif not self.tile[0] and not self.tile[1]:
                self.grid = self._grid_untiled.copy()
        elif self.shape_grid is None:
            self.grid = None
        if self.shape_image is not None:
            self.update_points()
    
    def update_points(self):
        if self.shape_grid is not None:
            if self.shape_image is not None:
                if not hasattr(self, "_points_unwarped"):
                    self._points_unwarped = gen_points(self.shape_grid, self.shape_image)
                if self.warp is not None:
                    self.points = warp_points(self._points_unwarped, self.warp)
                elif self.warp is None:
                    self.points = self._points_unwarped.copy()
            elif self.shape_image is None:
                self.points = None
        elif self.shape_image is None:
            self.points = None
        
    def image(self):
        return gen_image(self.grid, self.points)



if __name__ == "__main__":


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

    image = perlin_noise(shape_grid, shape_image)


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