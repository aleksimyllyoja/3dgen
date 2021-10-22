import numpy as np
import itertools
from math import *
from functools import reduce
from scipy.special import comb
from scipy.interpolate import splev
from math import *
from sdf import *

def bernstein(i, n, t):
    return comb(n, i)*(t**(n-i))*(1-t)**i

def _bpns(ps, n=100):
    return np.array([
        bernstein(i, len(ps)-1, np.linspace(0.0, 1.0, n))
        for i in range(0, len(ps))
    ])

def bezier(ps, n=5):
    return np.stack(list(map(
        lambda _ps: _ps@_bpns(ps, n),
        list(zip(*reversed(ps)))
    )), axis=-1)

def bspline(cv, n=100, degree=3):
    count = cv.shape[0]
    degree = np.clip(degree, 1, count-1)

    kv = np.array(
        [0]*degree+list(range(count-degree+1))+[count-degree]*degree,
        dtype='int'
    )

    u = np.linspace(0, (count-degree), n)

    return np.array(splev(u, (kv, cv.T, degree))).T

def spherical_coordinates(v):
    r = np.linalg.norm(v)
    return (r, atan2(v[1], v[0]), acos(v[2]/r))

def cartesian_coordinates(r, theta, phi):
    return np.array([
        r*sin(theta)*cos(phi),
        r*sin(theta)*sin(phi),
        r*cos(theta)
    ])

def plot_lines_and_points(lines, points=[]):
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(projection='3d')

    for _points in points:
        ax.scatter(
            *tuple(zip(*_points)),
            marker='o',
            color=[np.random.uniform(0.1, 0.8) for x in range(3)]
        )

    for line in lines:
        ax.plot(
            *tuple(zip(*line)),
            color=[np.random.uniform(0.1, 0.8) for x in range(3)]
        )

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.show()

def non_parallel_vector(v):
    if np.cross(np.array([1.0, 0.0, 0.0]), v).any():
        return np.array([1.0, 0.0, 0.0])
    else:
        np.array([0.0, 1.0, 0.0])

def perpendicular_vector(v):
    v /= np.linalg.norm(v)
    x = non_parallel_vector(v)
    x -= x.dot(v) * v
    return x/np.linalg.norm(x)

def rotation_matrix(axis, theta):
    axis = axis / math.sqrt(np.dot(axis, axis))
    a = math.cos(theta / 2.0)
    b, c, d = -axis * math.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])

def split_count(p0, p1, split_length):
    return floor(np.linalg.norm(p1-p0)/split_length)

def split_line(p0, p1, step):
    return [p0+(p1-p0)*k for k in np.arange(0, 1+step, step)]

def length(ps):
    return sum([np.linalg.norm(p1-p0) for p0, p1 in zip(ps, ps[1:])])

def segmented_line(p0, p1, point_distance=0.5):
    return split_line(p0, p1, split_count(p0, p1, point_distance))

def with_accumulated_length_fractions(path):
    return [(0, path[0])]+list(
        zip(
            itertools.accumulate([np.linalg.norm(p1-p0)/length(path) for p0, p1 in zip(path, path[1:])]),
            path
        )
    )

def circle_point(p, axis, radius, lateral_angle, vertical_angle):
    _pp = np.dot(rotation_matrix(axis, lateral_angle), perpendicular_vector(axis))
    _pp = np.dot(rotation_matrix(np.cross(_pp, axis), vertical_angle), _pp)

    return _pp/(np.linalg.norm(_pp)/radius)

def gen_mods(path, f_variation):
    return [(np.array([0,0,0]))]+[
        f_variation(p, l_acc)
        for l_acc, p in with_accumulated_length_fractions(path)[1:-1]
    ]+[(np.array([0,0,0]))]

def vary_path(path, mods):
    return np.array([p+m for p, m in zip(path, mods)])

def line_to_curvy_path(
    p0, p1,
    division = 0.2,
    f_variation = lambda point, l_acc: np.array([0,0,0]),
    n = 200
):
    line = split_line(p0, p1, division)
    return bspline(vary_path(line, gen_mods(line, f_variation)), n)

def constant(v):
    return lambda *args, **kwargs: v

def _exponential_scale(x, min, max):
    return exp(x*(log(max)-log(min))+log(min))

def exponential_scale(min, max):
    return lambda x: _exponential_scale(x, min, max)

def reversed_exp_scale(min, max):
    return lambda x: _exponential_scale(1-x, min, max)

def sin_bell(min, max):
    return lambda x: sin(x*pi)*(max+min)-min

def flatten(t):
    return [item for sublist in t for item in sublist]

def save_stl(sdf, samples=2**16, name_prefix="model", open_after_saving=False):
    from datetime import datetime

    postfix = datetime.now().strftime('%d-%m-%Y_%H-%M')
    filename = f'out/{name_prefix}_{postfix}.stl'

    print(f"SAVING {filename}")
    sdf.save(filename, samples=samples)

    if open_after_saving:
        import subprocess
        subprocess.run(["f3d", filename])
