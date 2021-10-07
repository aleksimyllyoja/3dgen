import matplotlib.pyplot as plt
import numpy as np
import itertools
from scipy.special import comb
from math import *
import sys
from sdf import *
import os

DEBUG = bool(os.environ.get('DEBUG'))
if(DEBUG): print("\nDEBUG\n")

#np.random.seed(19680801)

def vec(*ps):
    return np.array(ps, dtype=np.float64)

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

def spherical_coordinates(v):
    r = np.linalg.norm(v)
    return (r, atan2(v[1], v[0]), acos(v[2]/r))

def cartesian_coordinates(r, theta, phi):
    return np.array([
        r*sin(theta)*cos(phi),
        r*sin(theta)*sin(phi),
        r*cos(theta)
    ])

def plot_lines_and_points(lines, points=None):

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(projection='3d')

    if points is not None:
        xs, ys, zs = tuple(zip(*points))
        ax.scatter(xs, ys, zs, marker='o')

    for line in lines:
        color = [np.random.uniform(0, 0.6) for x in range(3)]
        xs, ys, zs = tuple(zip(*line))
        ax.plot(xs, ys, zs, color=color)

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

def split_line(p0, p1, n):
    return [p0+(p1-p0)/n*i for i in range(n+1)] if n>0 else [p0, p1]

def circle_point(p, radius, lateral_angle, vertical_angle):
    _pp = np.dot(
        rotation_matrix(p, lateral_angle),
        perpendicular_vector(p)
    )
    _pp = np.dot(
        rotation_matrix(np.cross(p, _pp), vertical_angle),
        _pp
    )

    return _pp/(np.linalg.norm(_pp)/radius)

def length(ps):
    return sum([np.linalg.norm(p1-p0) for p0, p1 in zip(ps, ps[1:])])

def exponential_scale(x, min, max):
    return exp(x*(log(max)-log(min))+log(min))

def segmented_line(p0, p1, point_distance=0.5):
    return split_line(p0, p1, split_count(p0, p1, point_distance))

def vary_line(
    ps,
    f_radius=lambda: np.uniform(0, 0.2),
    f_angle=lambda: np.random.uniform(0, 2*pi)):

    line = [ps[0]]
    for _p0, _p1 in zip(ps[0:], ps[1:-1]):
        _p3 = circle_point(
            _p1-_p0,
            f_radius(),
            f_angle(),
            0
        )
        line.append(_p1+_p3)

    line.append(ps[-1])
    return np.array(line)

def plant_stem(p0, p1):
    control_points = vary_line(
        segmented_line(p0, p1, point_distance=0.02),
        lambda: np.random.uniform(0, 0.2)
    )
    return bezier(control_points, n=100)

def plant(p0, p1):
    l0 = plant_stem(p0, p1)
    lines = []

    branch_distance = 0.3
    last_branch = 0
    cumulative_length = 0
    for p0, p1 in zip(l0[0:], l0[1:-1]):
        cumulative_length += np.linalg.norm(p1-p0)
        if cumulative_length > last_branch + branch_distance:

            last_branch = cumulative_length

            branch_lateral_angle = np.random.uniform(0, 2*pi)
            branch_vertical_angle = np.random.uniform(pi, 2*pi)
            branch_length = np.random.uniform(0.4, 0.9)

            p2 = circle_point(
                p1-p0,
                branch_length,
                branch_lateral_angle,
                branch_vertical_angle
            )
            branch_base_line = segmented_line(p1, p1+p2, point_distance=0.05)
            bcps = vary_line(
                branch_base_line,
                lambda: np.random.uniform(0, 0.05)
            )
            lines.append(bezier(bcps, n=20))

    return (l0, lines)

#lines = plant(vec(0, 0, -1), vec(0, 0, 1))
#plot_lines_and_points(lines)

#def render_lines():

from sdf import *

filename = 'blobby.stl'
stem, lines = plant(vec(0, 0, -1), vec(0, 0, 1))

def sdf_line(l, base=None):
    blob = base or sphere(0.01, l[0])
    total_length = length(l)
    cumulative_length = 0
    for p0, p1 in zip(l, l[1:]):
        cumulative_length += np.linalg.norm(p1-p0)

        x = 1 - cumulative_length/total_length
        k = exponential_scale(x, 0.1, 0.5)
        #print(k)
        blob = blob | sphere(0.01, p1).k(k)

    return blob

blob = sdf_line(stem)

for l in lines:
    total_length = length(l)
    cumulative_length = 0
    for p0, p1 in zip(l, l[1:]):
        cumulative_length += np.linalg.norm(p1-p0)
        x = 1 - cumulative_length/total_length
        k = exponential_scale(x, 0.02, 0.2)
        blob = blob | sphere(0.05, p1).k(k)

#blob = blob.erode(0.01)
#blob = stem_blob | branch_blob.k(0.5)

ground = box((3, 3, 0.5), stem[0])
blob = blob-ground

blob.save(filename, samples=2**24)

import subprocess
subprocess.run(["open", filename])
