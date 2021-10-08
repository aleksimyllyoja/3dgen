import argparse
import matplotlib.pyplot as plt
import numpy as np
import itertools
from functools import reduce
from scipy.special import comb
from math import *
from sdf import *
from time import time

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

def with_accumulated_length_fractions(path):
    return list(zip(itertools.accumulate([
        np.linalg.norm(p1-p0)/length(path)
        for p0, p1 in zip(path, path[1:])
    ]), list(zip(path, path[1:]))))

def vary_path(
    path,
    f_radius=lambda x: np.uniform(0, 0.2),
    f_angle=lambda: np.random.uniform(0, 2*pi)
):
    return np.array([path[0]]+[
        p1 + circle_point(
            p1-p0,
            f_radius(l_acc),
            f_angle(),
            0
        )
        for l_acc, (p0, p1) in with_accumulated_length_fractions(path)[:-1]
    ]+[path[-1]])

def curvy_path(
    p0,
    p1,
    variation_division = 0.02,
    f_variation = lambda x: np.uniform(0, 0.2),
    f_angle = lambda: np.random.uniform(0, 2*pi),
    n = 200
):
    return bezier(
        vary_path(
            segmented_line(p0, p1, point_distance=variation_division),
            f_radius=f_variation,
            f_angle=f_angle
        ),
        n=n
    )

def curvy_path_to_direction(
    p0,
    p1,
    length,
    lateral_angle,
    vertical_angle,
    variation_division,
    f_variation
):
    return curvy_path(
        p1, p1+circle_point(p1-p0, length, lateral_angle, vertical_angle),
        variation_division=variation_division,
        f_variation=f_variation
    )

def constant(v):
    return lambda *args, **kwargs: v

def reversed_exp_scale(min, max):
    return lambda x: exponential_scale(1-x, min, max)

def sin_bell(min, max):
    return lambda x: sin(x*pi)*(max+min)-min

def sdf_path(l, f_brush_size, f_k):

    total_length = length(l)
    cumulative_length = 0
    blob = sphere(f_brush_size(0), l[0])

    for p0, p1 in zip(l, l[1:]):
        cumulative_length += np.linalg.norm(p1-p0)
        x = cumulative_length/total_length
        blob = blob | sphere(f_brush_size(x), p1).k(f_k(x))

    return blob

def save_stl(sdf, open_after_saving=False):
    from datetime import datetime

    postfix = datetime.now().strftime('%d-%m-%Y_%H-%M')
    filename = f'out/blobby_{postfix}.stl'

    print(f"SAVING {filename}")
    print(f"SAMPLES {args.samples}")
    sdf.save(filename, samples=eval(args.samples.replace("^", "**")))

    if open_after_saving:
        import subprocess
        subprocess.run(["f3d", filename])

def gen_branches(
    path,
    f_branch_count = lambda *args: int(np.random.uniform(0, 1.05)),
    f_branch_length = reversed_exp_scale(0.4, 1.5),
    f_branch_crookedness = lambda l_acc: lambda x: np.random.uniform(0, 0.1),
    f_variation_division = constant(0.05),
    f_lateral_angle = lambda *args: np.random.uniform(0, 2*pi),
    f_vertical_angle = lambda *args: np.random.uniform(pi, 2*pi)
):
    return [
        (l_acc, curvy_path_to_direction(
            p0, p1, f_branch_length(l_acc),
            lateral_angle=f_lateral_angle(l_acc),
            vertical_angle=f_vertical_angle(l_acc),
            variation_division=f_variation_division(l_acc),
            f_variation=f_branch_crookedness(l_acc)
        ))
        for l_acc, (p0, p1) in with_accumulated_length_fractions(path)
        for i in range(f_branch_count(l_acc))
    ]

def sdf_from_path_tree(
    tree,
    f_brush_size = lambda level, **kwargs: constant(0.01),
    f_k_brush = lambda level, **kwargs: constant(0.01),
    f_k_paths = constant(0.02),
    parent_kwargs = {}
):
    f_args = dict(tree.get('kwargs'), **{'parent_kwargs': parent_kwargs})

    path_brush_size = f_brush_size(**f_args)
    parent_kwargs = {
        'f_brush_size': path_brush_size,
    }

    return reduce(
        lambda a, b: a | b.k(f_k_paths(**f_args)),
        [
            sdf_path(
                tree.get('path'),
                f_brush_size = f_brush_size(**f_args),
                f_k = f_k_brush(**f_args)
            )
        ] + [
            sdf_from_path_tree(child, f_brush_size, f_k_brush, f_k_paths, parent_kwargs)
            for child in tree.get('children')
        ]
    )

def tree_to_list_of_paths(tree):

    def _flatten(t):
        return [item for sublist in t for item in sublist]

    def _traverse(tree, paths=[]):
        if not tree.get('children'):
            return [tree.get('path')]
        return [tree.get('path')]+_flatten([_traverse(c) for c in tree.get('children')])


    return list(_traverse(tree))

def setup():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--samples', default='2^16',
                        help='SDF render samples count')
    parser.add_argument('--seed', type=int, default=int(time()),
                        help='SDF render samples count')
    parser.add_argument('--save', action='store_true', default=False)
    parser.add_argument('--lineplot', action='store_true', default=False)
    args = parser.parse_args()

    return args

args = setup()

print(f"RANDOM SEED {args.seed}")

np.random.seed(args.seed)

#####

stem = curvy_path(
    vec(0, 0, -1), vec(0, 0, 1),
    f_variation=lambda x: np.random.uniform(0, sin_bell(0, 0.5)(x))
)

branches = gen_branches(
    stem,
    f_branch_count = lambda l_acc: (
        0.20 < l_acc and 0.9 > l_acc
        and bool(int(np.random.uniform(0, 1.02)))
    ) * np.random.randint(1, 4),
    f_branch_length = reversed_exp_scale(0.4, 1.5),
    f_branch_crookedness = lambda l_acc:
        lambda x: np.random.uniform(0, reversed_exp_scale(0.05, 0.2)(l_acc)),
)

path_tree = {
    'path': stem,
    'kwargs': {
        'level': 0,
    },
    'children': list(map(lambda b: ({
        'path': b[1],
        'kwargs': {
            'l_acc_parent': b[0],
            'level': 1
        },
        'children': []
    }), branches))
}

"""
path_tree['children'][0]['children'] = list(map(lambda b: ({
    'path': b[1],
    'kwargs': {
        'l_acc_parent': b[0],
        'level': 2
    },
    'children': []
}), gen_branches(
    path_tree['children'][0]['path'],
    f_branch_count = lambda l_acc: (
        0.20 < l_acc and 0.9 > l_acc
        and bool(int(np.random.uniform(0, 1.02)))
    ) * np.random.randint(1, 4),
    f_branch_length = reversed_exp_scale(0.4, 1.5),
    f_branch_crookedness = lambda l_acc:
        lambda x: np.random.uniform(0, reversed_exp_scale(0.05, 0.2)(l_acc)),
)))
"""

paths = tree_to_list_of_paths(path_tree)
if args.lineplot: plot_lines_and_points(tree_to_list_of_paths(path_tree))

blob = sdf_from_path_tree(
    path_tree,
    f_brush_size = lambda level, **kwargs:
        reversed_exp_scale(0.02, 0.2) if level==0 else
        reversed_exp_scale(
            0.01,
            kwargs['parent_kwargs']['f_brush_size'](kwargs.get('l_acc_parent'))
        )
    ,
    f_k_brush = lambda level, **kwargs: [
        constant(0.01),
        reversed_exp_scale(0.01, 0.05),
        reversed_exp_scale(0.01, 0.05)
    ][level],
    f_k_paths = constant(0.02)
)

blob = blob & slab(z0=-1)

if args.save: save_stl(blob, open_after_saving=True)
