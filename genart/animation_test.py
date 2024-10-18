import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.signal import convolve2d

# def game_of_life(input_array):
#     output_array = np.zeros_like(input_array)
#     for index in np.ndindex((input_array.shape[0] - 2, input_array.shape[1] - 2)):
#         input_cell = input_array[index[0] + 1, index[1] + 1]
#         neighbours = input_array[index[0]:index[0] + 3, index[1]:index[1] + 3].sum() - input_cell
#         if neighbours in [2, 3]:
#             if input_cell:
#                 output_array[index[0] + 1, index[1] + 1] = 1
#             elif not input_cell and neighbours == 3:
#                 output_array[index[0] + 1, index[1] + 1] = 1
#     return output_array

def cellular_automaton(input_array, rule):
    kernel = np.ones((3, 3), dtype = int)
    kernel[1, 1] = 0 # Messing around with the kernel gives some cool patterns.
    convolved_array = convolve2d(input_array.astype(int), kernel, mode = "same", boundary = "wrap")
    return np.where(np.logical_or(np.logical_and(~input_array, np.isin(convolved_array, rule["B"])), np.logical_and(input_array, np.isin(convolved_array, rule["S"]))), True, False)

def voronoi(rng, shape, points = None, number_points = None, wrapped = False, return_points = False):
    if points is None:
        points = np.array([(y, x) for y, x in zip(rng.uniform(0, shape[0], size = number_points), rng.uniform(0, shape[1], size = number_points))])
    if wrapped:
        points_offset = np.concatenate([points + offset for offset in [[y_offset, x_offset] for y_offset in [-shape[0], 0, shape[0]] for x_offset in [-shape[1], 0, shape[1]]]], axis = 0)
    coordinates = np.array(np.meshgrid(np.arange(shape[1]), np.arange(shape[0])))
    if wrapped:
        regions = np.array([np.linalg.norm(coordinates - point[::-1, None, None], axis = 0) for point in points_offset]).argmin(0)%points.shape[0]
    elif not wrapped:
        regions = np.array([np.linalg.norm(coordinates - point[::-1, None, None], axis = 0) for point in points]).argmin(0)
    return (regions, points) if return_points else regions

rng = np.random.default_rng(3)

image_shape = (64, 128)
display_shape = (512, 1024)
alive_dead_ratio = 0.25
frames_per_second = 60

# rule = {"B" : [3], "S" : [2, 3]}
# rule = {"B" : [4, 6, 7, 8], "S" : [3, 5, 6, 7, 8]}
# rule = {"B" : [3, 5, 6, 7, 8], "S" : [5, 6, 7, 8]}
# rule = {"B" : [1], "S" : [0, 1, 2, 3, 4, 5, 6, 7, 8]}
# rule = {"B" : [0, 1, 3, 5, 6], "S" : [0, 1, 2, 3, 4, 5]}
# rule = {"B" : [3, 4, 5], "S" : [4, 5, 6, 7]}
# rule = {"B" : [3, 5, 6, 7, 8], "S" : [4, 6, 7, 8]}
# rule = {"B" : [3, 5], "S" : [2, 3, 4, 5, 7, 8]}
# rule = {"B" : [5, 6, 7, 8], "S" : [4, 5, 6, 7, 8]}
rule = {"B" : [3, 5, 7, 8], "S" : [2, 4, 6, 7, 8]}
# rule = {"B" : [2], "S" : []}
# rule = {"B" : [2, 3, 4], "S" : []}
# rule = {"B" : [3, 6, 7, 8], "S" : [3, 4, 6, 7, 8]}

init_array = rng.choice([0, 1], size = image_shape, p = [1 - alive_dead_ratio, alive_dead_ratio])
# init_array = np.pad(init_array, ((64, 64), (128, 128)))

# # Simple one-rule animation
# figure, axis = plt.subplots(1, 1, figsize = (8, 4), layout = "constrained")
# axis.axis("off")
# image = axis.imshow(init_array.astype(bool), cmap = "Greys", vmin = False, vmax = True, interpolation = "none")
# def animate(index):
#     image.set_array(cellular_automaton(image.get_array(), rule))
#     return [image]
# animation = FuncAnimation(figure, animate, interval = 1000/frames_per_second, blit = True, cache_frame_data = False)
# plt.show()

from scipy.signal import fftconvolve
from skimage.transform import resize

# Slowly building sum of automata
array_automaton = init_array.copy()
array_automaton_sum = array_automaton.astype(float)
array_display = resize(array_automaton_sum, display_shape, order = 0)
figure, axis = plt.subplots(1, 1, figsize = (display_shape[1]/96, display_shape[0]/96), layout = "constrained")
axis.axis("off")
image = axis.imshow(array_display, vmin = 0, vmax = 1, cmap = "Greys_r", interpolation = "none")
def animate(index):
    global array_automaton, array_automaton_sum
    array_automaton = cellular_automaton(array_automaton, rule)
    array_automaton_sum += array_automaton.astype(float)
    array_display = resize(array_automaton_sum, display_shape, order = 0)
    image.set_array(array_display)
    image.set_clim(vmin = array_display.min(), vmax = array_display.max())
    array_automaton_sum *= 0.5
    array_automaton_sum += 0.15*fftconvolve(array_automaton_sum, np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]]), mode = "same")
    return [image]
animation = FuncAnimation(figure, animate, interval = 1000/frames_per_second, blit = True, cache_frame_data = False)
plt.show()



# rule_0 = {"B" : [3], "S" : [2, 3]}
# rule_1 = {"B" : [3, 5], "S" : [2, 3, 4, 5, 7, 8]}
# array_automaton = init_array.copy()
# points = np.array([(y, x) for y, x in zip(rng.uniform(0, 64, size = 8), rng.uniform(0, 128, size = 8))])
# angles = rng.uniform(0, 2*np.pi, size = points.shape[0])
# speeds = rng.uniform(0, 0.5, size = points.shape[0])
# velocities = np.array([(speed*np.sin(angle), speed*np.cos(angle)) for angle, speed in zip(angles, speeds)])
# figure, axis = plt.subplots(1, 1, figsize = (8, 4), layout = "constrained")
# axis.axis("off")
# image = axis.imshow(array_automaton, vmin = array_automaton.min(), vmax = array_automaton.max(), cmap = "Greys", interpolation = "none")
# def animate(index):
#     global array_automaton, points
#     points += velocities
#     points %= np.array(array_automaton.shape)
#     regions = voronoi(rng, array_automaton.shape, points, wrapped = True)
#     array_automaton[np.where(regions%2, True, False)] = cellular_automaton(array_automaton, rule_0)[np.where(regions%2, True, False)]
#     array_automaton[np.where(regions%2, False, True)] = cellular_automaton(array_automaton, rule_1)[np.where(regions%2, False, True)]
#     image.set_array(array_automaton)
#     image.set_clim(vmin = array_automaton.min(), vmax = array_automaton.max())
#     return [image]
# animation = FuncAnimation(figure, animate, interval = 1000/frames_per_second, blit = True, cache_frame_data = False)
# plt.show()