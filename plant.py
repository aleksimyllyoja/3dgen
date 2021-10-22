import argparse
import numpy as np
import itertools
from functools import reduce
from scipy.special import comb
from scipy.interpolate import splev
from math import *
from sdf import *
from time import time
from utils import *

def sdf_path(l, f_brush_size, f_k):
    total_length = length(l)
    cumulative_length = 0
    blob = sphere(f_brush_size(0), l[0])

    for p0, p1 in zip(l, l[1:]):
        cumulative_length += np.linalg.norm(p1-p0)
        x = cumulative_length/total_length
        blob = blob | sphere(f_brush_size(x), p1).k(f_k(x))

    return blob

def sdf_from_path_tree(
    tree,
    f_brush_size = lambda level, **kwargs: constant(0.01),
    f_k_brush = lambda level, **kwargs: constant(0.01),
    f_k_paths = constant(0.02),
    parent_kwargs = {}
):
    f_args = dict(tree.get('attr'), **{'parent_kwargs': parent_kwargs})

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

def tree_to_paths(tree):
    def _traverse(tree, paths=[]):
        if not tree.get('children'):
            return [tree.get('path')]
        return [tree.get('path')]+flatten([_traverse(c) for c in tree.get('children')])

    return list(_traverse(tree))

def gen_tree(p0, p1, max_depth=2, **kwargs):

    def gen_branch_lines_and_points(path, _level):
        return [
            (
                l_acc,
                p,
                p+circle_point(
                    p,
                    axis=p1-p0,
                    radius=kwargs.get('f_branch_length')(l_acc),
                    lateral_angle=np.random.uniform(0, 2*pi),
                    vertical_angle=0 #np.random.uniform(pi, 2*pi)
                )
            )
            for l_acc, p in with_accumulated_length_fractions(path)
            for i in range(kwargs.get('f_branch_count')(l_acc, _level))
        ]

    def _gen_tree(_tree={}, **kwargs):
        _level = _tree.get('attr').get('level')+1
        if _tree.get('attr').get('level') >= max_depth: return _tree

        _tree['children'] = [
            _gen_tree({
                'path': line_to_curvy_path(
                    _p0, _p1,
                    f_variation = kwargs.get('f_variation')(_level, _p1-_p0)
                ),
                'attr': {
                    'level': _level,
                    'l_acc_parent': l_acc
                },
                'children': []
            }, **kwargs)

            for l_acc, _p0, _p1 in gen_branch_lines_and_points(_tree.get('path'), _level)
        ]

        return _tree

    return _gen_tree(
        **kwargs,
        _tree = {
            'path': line_to_curvy_path(
                p0, p1,
                division = 0.2,
                f_variation = kwargs.get('f_variation')(0, p1-p0)
            ),
            'attr': {
                'level': 0,
            },
            'children': []
        }
    )

def setup():
    parser = argparse.ArgumentParser()
    parser.add_argument('--samples', default='2^16',
                        help='SDF render samples count')
    parser.add_argument('--seed', type=int,
                        default=np.random.randint(0, 2**32-1),
                        help='random seed')
    parser.add_argument('--save', action='store_true', default=False)
    parser.add_argument('--lineplot', action='store_true', default=False)
    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = setup()

    print(f"RANDOM SEED {args.seed}")

    np.random.seed(args.seed)

    p0 = np.array((0.0, 0.0, -1.0))
    p1 = np.array((0.0, 0.0, 1.0))

    path_tree = gen_tree(
        p0, p1,
        max_depth = 2,
        f_branch_count = lambda l_acc, level:
            np.random.randint(1, 4 if level == 0 else 2) * (
                0.20 < l_acc and
                0.99 > l_acc and
                0.98 < np.random.random()
            ),
        f_branch_length = reversed_exp_scale(0.6, 0.8),
        f_variation = lambda level, axis: lambda point, l_acc:
            circle_point(
                point,
                axis,
                np.random.uniform(0.0, 0.2),
                np.random.uniform(0, 2*pi),
                np.random.uniform(0, 2*pi)
            )
        ,
        f_variation_division = exponential_scale(0.2, 0.4),
    )

    if args.lineplot: plot_lines_and_points(tree_to_paths(path_tree))

    blob = sdf_from_path_tree(
        path_tree,

        f_brush_size = lambda level, **kwargs:
            reversed_exp_scale(0.02, 0.2) if level==0 else
            reversed_exp_scale(
                0.01,
                kwargs['parent_kwargs']['f_brush_size'](kwargs.get('l_acc_parent'))*0.6
            ),

        f_k_brush = lambda level, **kwargs: [
                constant(0.01),
                reversed_exp_scale(0.01, 0.05),
                reversed_exp_scale(0.01, 0.05)
            ][level],

        f_k_paths = constant(0.02)
    )

    blob = blob & slab(z0=-1)

    if args.save:
        print(f"SAMPLES {args.samples}")
        save_stl(
            blob,
            samples=eval(args.samples.replace("^", "**")),
            name_prefix="plant",
            open_after_saving=True
        )
