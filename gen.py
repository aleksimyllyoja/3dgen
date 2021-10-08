import argparse
import matplotlib.pyplot as plt
import numpy as np
import itertools
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

def vary_line(
    ps,
    f_radius=lambda x: np.uniform(0, 0.2),
    f_angle=lambda: np.random.uniform(0, 2*pi)
):

    line = [ps[0]]
    total_length = length(ps)
    cumulative_length = 0
    for _p0, _p1 in zip(ps, ps[1:-1]):
        cumulative_length += np.linalg.norm(_p1-_p0)
        _p3 = circle_point(
            _p1-_p0,
            f_radius(cumulative_length/total_length),
            f_angle(),
            0
        )
        line.append(_p1+_p3)

    line.append(ps[-1])
    return np.array(line)

def curvy_path(
    p0,
    p1,
    variation_division=0.02,
    f_variation=lambda x: np.uniform(0, 0.2),
    f_angle=lambda: np.random.uniform(0, 2*pi),
    n=200
):
    return bezier(
        vary_line(
            segmented_line(p0, p1, point_distance=variation_division),
            f_radius=f_variation,
            f_angle=f_angle
        ),
        n=n
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

# ==========

def plant(p0, p1, f_branch_length, f_max_crookedness):
    l0 = curvy_path(
        p0,
        p1,
        f_variation=lambda x: np.random.uniform(0, sin_bell(0, 0.5)(x))
    )
    lines = []

    branch_distance = 0.20
    branch_distance_threshold = 0.20
    branch_distance_top_threshold = 0.40
    last_branch = 0
    cumulative_length = 0
    total_length = length(l0)

    for p0, p1 in zip(l0[0:], l0[1:-1]):

        cumulative_length += np.linalg.norm(p1-p0)

        can_branch = cumulative_length-branch_distance_threshold > last_branch + branch_distance
        not_too_high = cumulative_length < total_length-branch_distance_top_threshold
        if can_branch and not_too_high:
            x = cumulative_length/total_length

            last_branch = cumulative_length

            for i in range(np.random.randint(1, 4)):
                branch_lateral_angle = np.random.uniform(0, 2*pi)
                branch_vertical_angle = np.random.uniform(pi, 2*pi)
                branch_crookedness = lambda x: np.random.uniform(0, f_max_crookedness(x))

                p2 = circle_point(
                    p1-p0,
                    f_branch_length(x),
                    branch_lateral_angle,
                    branch_vertical_angle
                )

                branch_curve = curvy_path(
                    p1,
                    p1+p2,
                    variation_division=0.05,
                    f_variation=branch_crookedness
                )
                #lines.append(bcps)
                lines.append((x, branch_curve))

    return (l0, lines)

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--samples', default='2^16',
                    help='SDF render samples count')
parser.add_argument('--seed', type=int, default=int(time()),
                    help='SDF render samples count')
parser.add_argument('--save', action='store_true', default=False)
parser.add_argument('--lineplot', action='store_true', default=False)
args = parser.parse_args()

print(f"RANDOM SEED {args.seed}")

np.random.seed(args.seed)

stem, lines = plant(
    vec(0, 0, -1), vec(0, 0, 1),
    f_branch_length = reversed_exp_scale(0.4, 1.5),
    f_max_crookedness = reversed_exp_scale(0.05, 0.2)
)

if args.lineplot:
    plot_lines_and_points([stem]+[l for (x, l) in lines])

f_stem_brush_size = reversed_exp_scale(0.02, 0.2)

blob = sdf_path(
    stem,
    f_brush_size = f_stem_brush_size,
    f_k = constant(0.01), #reversed_exp_scale(0.01, 0.05)
)

for (x, l) in lines:
    branch_blob = sdf_path(
        l,
        f_brush_size = reversed_exp_scale(0.01, f_stem_brush_size(x)*0.7),
        f_k = reversed_exp_scale(0.01, 0.05)
    )

    blob = blob | branch_blob.k(0.02)

blob = blob & slab(z0=-1)

if args.save:
    save_stl(blob, open_after_saving=True)
