import random
import math
import matplotlib.pyplot as plt
from enum import Enum
import numpy as np
from scipy.interpolate import BSpline, splprep, splev
import copy


class Point:
    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y

    def __str__(self):
        return '({x},{y})'.format(x=self.x, y=self.y)

    def get_point(self):
        point = (self.x, self.y)
        return point

    def __getitem__(self, key):
        return self.x if not key else self.y


class Map:
    def __init__(self, starting_point, resolution=(1280, 720), seed=None):
        self.points = []
        self.append(starting_point)
        self.resolution = resolution
        self.seed = seed

    def set_with_arrays(self, x_array, y_array):
        self.points.clear()
        data = zip(x_array, y_array)
        self.points = [Point(x, y) for x, y in data]

    def append(self, point):
        self.points.append(point)

    def __getitem__(self, key):
        return self.points[key]

    def get_data(self):
        data = [(point.x, point.y) for point in self.points]
        return data

    def X(self):
        xs = [point.x for point in self.points]
        return xs

    def Y(self):
        ys = [point.y for point in self.points]
        return ys

    def __len__(self):
        return len(self.points)

    def __delitem__(self, index):
        del self.points[index]

    def is_crossing(self, i, j):
        """Function that checks if two lines ((X[i],Y[i]), (X[i+1], Y[i+1])), ((X[j],Y[j]), (X[j+1],Y[j+1])) intersect"""
        if i == j:
            return False
        try:
            return intersect(self.points[i], self.points[i + 1], self.points[j], self.points[j + 1])
        except IndexError:
            print(f'i = {i}, j = {j}, size = {len(self)}')

    def exterminate_loops(self, step, y_res, ran=40):
        """Function that gets rid of most of the loops in map so as to make it a little less crazy"""
        i = 0
        size = len(self.points)
        while i < size - 1:
            i = size - 2 if i >= size - 1 else i
            left_limit = i - ran if i >= ran else 0
            right_limit = i + ran if i + ran < size else size - 2
            for j in range(left_limit + 1, right_limit):
                j = size - 2 if j >= size - 1 else j
                if self.is_crossing(i, j):
                    del self.points[i + 1]
                    point = next_point(self.points[size - 2], step, math.pi * 3 / 4, y_res)
                    self.append(point)
            i += 1

    def save_to_file(self, filename='test_map.png', fill=False):
        plt.clf()
        plt.figure(figsize=(self.resolution[0] / 100, self.resolution[1] / 100))
        plt.xlim(0.0, self.resolution[0])
        plt.ylim(0.0, self.resolution[1])
        if fill:
            plt.fill_between(self.X(), self.Y())
        plt.plot(self.X(), self.Y())
        plt.savefig(filename, dpi=100)

    def show_map(self, fill=False):
        plt.clf()
        plt.figure(figsize=(self.resolution[0] / 100, self.resolution[1] / 100))
        plt.xlim(0.0, self.resolution[0])
        plt.ylim(0.0, self.resolution[1])
        if fill:
            plt.fill_between(self.X(), self.Y())
        plt.plot(self.X(), self.Y())
        plt.show()


class Difficulty(Enum):
    """Enum for Difficulty level"""
    PATHETIC = 0
    EASY = 1
    MEDIUM = 2
    HARD = 3
    REALLY_HARD = 4
    WTF = 5


def intersect(A, B, C, D):
    """Helper of is_crossing function
    Returns True if line segments AB and CD intersect"""
    ccw = lambda A, B, C: (C.y - A.y) * (B.x - A.x) > (B.y - A.y) * (C.x - A.x)
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)


def next_point(prev_point, step, alpha, y_res):
    """Function returning next random point for the generalized map"""
    y_range = 2 * step * math.tan(alpha / 2)
    ymin = prev_point.y - y_range / 2
    ymin = ymin if ymin >= 0 else 0
    ymax = prev_point.y + y_range / 2
    ymax = ymax if ymax < y_res - 20 else y_res - 20
    p = random.random()
    if p >= 0.3:
        x = prev_point.x + step
    else:
        x = prev_point.x - step if prev_point.x > step else prev_point.x + step
    y = random.uniform(ymin, ymax)
    point = Point(x, y)
    return point


def prepare_map(resolution, step, angle_range, starting_y, seed):
    """Function that prepares real map - interpolates random points generated by function next_point
        returns interpolated and smoothed map in X and Y arrays
    """
    x_res = resolution[0]
    y_res = resolution[1]

    x = 0
    y = random.randrange(y_res / 4, y_res * 3 / 5) if not starting_y else starting_y
    point = Point(x, y)
    game_map = Map(point, resolution, seed)

    while point.x < x_res:
        point = next_point(point, step, angle_range, y_res)
        game_map.append(point)

    game_map.exterminate_loops(step, y_res)
    game_map.exterminate_loops(step, y_res)
    game_map.exterminate_loops(step, y_res)

    tck, u = splprep([game_map.X(), game_map.Y()], s=0)
    x_array = np.linspace(0, 1, x_res)
    x_array, y_array = splev(x_array, tck, der=0)

    game_map.set_with_arrays(x_array, y_array)

    return game_map


def get_level_parameters(diff_level, x_res):
    """Function that prepares parameters according to difficulty level of map
        returns X and Y ready to go
    """
    step = None
    angle = None

    if diff_level == Difficulty.PATHETIC:
        step = 2
        angle_range = 0

    elif diff_level == Difficulty.EASY:
        step = x_res / 10
        angle_range = math.pi / 4

    elif diff_level == Difficulty.MEDIUM:
        step = x_res / 20
        angle_range = math.pi * 3 / 4

    elif diff_level == Difficulty.HARD:
        step = x_res / 100
        angle_range = 15 * math.pi / 18

    elif diff_level == Difficulty.REALLY_HARD:
        step = x_res / 120
        angle_range = 16 * math.pi / 18

    elif diff_level == Difficulty.WTF:
        step = x_res / 150
        angle_range = 16 * math.pi / 18

    return step, angle_range


def generate_map(seed=None, diff_level=Difficulty.PATHETIC, starting_y=None, resolution=(1280, 720)):
    """Function to generate map
    parameters - map seed generated before, difficulty level from Difficulty Enum, starting y position, resolution
    returns data in format [(x1,y1), (x2,y2), .....] as points coordinates
    """
    if seed:
        random.setstate(seed)
    else:
        seed = random.getstate()

    step, angle_range = get_level_parameters(diff_level, resolution[0])
    game_map = prepare_map(resolution, step, angle_range, starting_y, seed)

    return game_map, seed
    # return game_map.get_data(), seed

# USE EXAMPLES
map1, seed1 = generate_map()
# map1.save_to_file()
map2, seed2 = generate_map(diff_level = Difficulty.REALLY_HARD)
# map2.save_to_file('test_really_hard.png')
# data3, seed3 = generate_map(diff_level = Difficulty.EASY)
# save_map_to_file(data3, 'test_EASY.png')
# data, seed = generate_map(diff_level = Difficulty.HARD, starting_y = 120)
# save_map_to_file(data, 'test_hard.png')
# save_map_to_file(data, 'filled_hard.png', fill = True)
# data, seed = generate_map(seed, diff_level = Difficulty.HARD, starting_y = 120)
# save_map_to_file(data, 'copy_hard.png')

map1.show_map(fill=True)
map2.show_map()