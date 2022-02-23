from sdf import *
from utils import *

from copy import copy
from termcolor import colored

import argparse

import inspect
import numpy as np
from copy import copy

np_random_uniform = copy(np.random.uniform)

OVERWRITES = {}

def _uniform(a, b):
    fname = inspect.stack()[1].code_context[0].split("=")[0].strip()
    v = OVERWRITES.get(fname, np_random_uniform(a, b))
    print(f"{fname: <32} {v:.4f} [{a:.3f}, {b:.3f}]")
    return v

np.random.uniform = _uniform

def eye(opening = pi/9, scale=0.15):
    f = sphere() & slab(z0=0)
    f2 = copy(f).rotate(-opening, X).translate((0, -0.2, 0))
    f = f.rotate(pi, X).rotate(opening, X)
    f = f | f2
    f = f | sphere(0.8)

    return f.scale(scale)

def eyes(pd=0.5, scale=0.15, left_opening=pi/9, right_opening=pi/9):
    eye_1 = eye(left_opening, scale).translate((-pd/2, 0, 0))
    eye_2 = eye(right_opening, scale).translate((pd/2, 0, 0))

    return eye_1 | eye_2

def eye_cavities(size, cavity_distance=0.7):
    eyec_1 = ellipsoid(size).translate((-cavity_distance/2, 0, 0))
    eyec_1 &= plane(-X).translate((-cavity_distance/2, 0, 0)).k(0.2)

    eyec_2 = ellipsoid(size).translate((cavity_distance/2, 0, 0))
    eyec_2 &= plane(X).translate((cavity_distance/2, 0, 0)).k(0.2)

    return eyec_1 | eyec_2.k(0.2)

def forehead(face_width=1):
    return sdf_path(bspline(np.array([
        (-1, 0.4, 0),
        (-0.5, 0, 0.2),
        (0, 0.5, -0.2),
        (0.5, 0, 0.2),
        (1, 0.4, 0),
    ]), n=100), f_brush_size=sin_bell(0.01, 0.1))

def lips(opening=0.9, upper_lip_thickness=0.1, lower_lip_thickness=0.1):
    upper_lip = sdf_path(bspline(np.array([
        (-1, 0.5, 0),
        (-0.25, 0, 0.2),
        (0, 0, 0),
        (0.25, 0, 0.2),
        (1, 0.5, 0)
    ]), n=50), f_brush_size=constant(upper_lip_thickness))

    lower_lip = sdf_path(bspline(np.array([
        (-1, 0.5, 0),
        (0, 0, -opening),
        (1, 0.5, 0)
    ]), n=50), f_brush_size=constant(lower_lip_thickness)).translate((0, 0, -0.1))

    upper_lip = upper_lip - ellipsoid((0.2, 0.2, 0.2)).translate((0, 0, 0.4)).k(0.2)

    return upper_lip | lower_lip

def nose(
    tip_size=0.9,
    nostril_size=0.2,
    nose_angle=pi/12,
    depth=0.4,
    hook_upper_strength=0.4,
    hook_lower_strength=0.4,
    nose_width=0.5,
    subnasal_angle=pi/16,
    subnasal_cut=-0.4,
):
    n1 = sdf_path(bspline(np.array([
        (0, 0.1-hook_upper_strength/10.0, 3),
        (0, -hook_upper_strength, 2.5),
        (0, -hook_lower_strength, 2),
        (0, 0, 0),
    ]), n=50), f_brush_size=exponential_scale(tip_size*0.5, tip_size))

    n2 = sdf_path(bspline(np.array([
        (0, depth, 3),
        (0, depth, 0),
        (-nose_width/2, depth, 0.2),
    ]), n=50), f_brush_size=exponential_scale(0.2, nostril_size*tip_size))

    n3 = sdf_path(bspline(np.array([
        (0, depth, 3),
        (0, depth, 0),
        (nose_width/2, depth, 0.2),
    ]), n=50), f_brush_size=exponential_scale(0.2, nostril_size*tip_size))

    filler = sdf_path(bspline(np.array([
        (0, 0.2-hook_upper_strength/10.0, 3),
        (0, -hook_upper_strength, 2.5),
        (0, -hook_lower_strength, 2),
        (0, 0.2, 0),
    ]), n=50), f_brush_size=exponential_scale(tip_size*0.5, tip_size))

    f_nose = (n1 | filler.k(0.2)) | (n2 | n3)

    f_nose &= plane(Z).translate((0, 0, subnasal_cut)).rotate(subnasal_angle, X).k(0.2)

    return f_nose.rotate(nose_angle, -X).scale(0.2)

def grid(objs):
    blob = objs[0]
    l = ceil(sqrt(len(objs)))
    for i in range(l):
        for j in range(l):
            if(i*l+j >= len(objs)):
                break

            blob = blob | objs[i*l+j].translate((3*j, 0, 5*i))

    return blob

def busto():
    max_head_width = 3
    jaw_width = np.random.uniform(0.25, 1.2)
    head_width = np.random.uniform(2, max_head_width)

    chin_width = np.random.uniform(0.1, 0.6)
    chin_depth = np.random.uniform(0, 0.3)
    chin_hardness = np.random.uniform(0.1, 0.45)

    jaw_cut_height = np.random.uniform(-0.2, 0.2)
    jaw_cut_hardness = np.random.uniform(0.2, 0.4)

    jaw = (
        rounded_cone(jaw_width, 1, 1.4-jaw_width) & slab(x0=0).k(0.7)
    ).translate((0, 0, 0.5*jaw_width)).scale((1.5, 1, 1)) & plane(-X).translate((1.5, 0, 0)).k(0.4)

    chin = sdf_path(
        bspline(np.array([
            (0.9, chin_width/2, 0),
            (0.2, chin_width/2.2, 0),
            (0, 0, 0),
            (0.2, -chin_width/2.2, 0),
            (0.9, -chin_width/2, 0),
        ]), n=100), f_brush_size=sin_bell(0.01, 0.02)
    ).translate((chin_depth, 0, 0.4*(1.2-jaw_width)))

    jaw |= chin.k(chin_hardness)

    head_ball = sphere(1.25).translate((1.0, 0, 1.5)) & slab(x0=0).k(0.6)

    f_head = (head_ball | jaw.k(0.3)).rotate(pi/2, Z)

    f_head = f_head & slab(z0=jaw_cut_height).k(jaw_cut_hardness)

    nose_tip_size = np.random.uniform(0.4, 1.5)
    f_head |= nose(
        tip_size=nose_tip_size,
        nostril_size=np.random.uniform(0.2, 0.9),
        nose_angle=np.random.uniform(pi/16, pi/10),
        depth=np.random.uniform(0.1, 1.0),
        hook_upper_strength=np.random.uniform(-0.5, 0.5),
        hook_lower_strength=np.random.uniform(-1.0, 1.4),
        nose_width=np.random.uniform(0.5, 1.7),
        subnasal_angle=np.random.uniform(pi/16, pi/6),
        subnasal_cut=np.random.uniform(-0.6, 0.8)
    ).translate((0, -0.2, 1.0)).k(0.1)

    f_head -= eye_cavities(
        (0.6, 0.2, 0.2),
        cavity_distance = nose_tip_size*0.4
    ).translate((0, 0, 1.5)).k(0.2)

    f_head |= forehead().scale(0.75).translate((0, 0.0, 1.7)).k(0.2)

    f_head = f_head - box((0.5, 10, 10), (-head_width/2.0, 0, 0)).k(0.4) - box((0.5, 10, 10), (head_width/2.0, 0, 0)).k(0.4)

    left_cheek_control = sphere(0.4).translate((-1.4-(max_head_width-head_width)/5.0, 0, 0.5))
    right_cheek_control = sphere(0.4).translate((1.4-(max_head_width-head_width)/5.0, 0, 0.5))

    f_head -= left_cheek_control.k(0.6)
    f_head -= right_cheek_control.k(0.6)

    f_head |= eyes(
        pd=np.random.uniform(0.6, 0.8+head_width/7.5),
        left_opening=np.random.uniform(pi/14, pi/4),
        right_opening=np.random.uniform(pi/14, pi/4),
        scale=np.random.uniform(0.15, 0.3)
    ).translate((0, 0.25, 1.4)).k(0.1)

    lip_depth = np.random.uniform(0.05, 0.1)
    lip_height = np.random.uniform(0.6, 0.8)

    f_lips = lips(
        opening = np.random.uniform(0, 1.2),
        upper_lip_thickness = np.random.uniform(0.03, 0.2),
        lower_lip_thickness = np.random.uniform(0.03, 0.2)
    ).scale(0.3).translate((0, -lip_depth, lip_height))

    f_head |= f_lips.k(0.1)

    # arg no rotations
    head_tilt = np.random.uniform(-pi/13, pi/13)
    f_head = f_head.rotate(head_tilt, X)
    #f_head = f_head.rotate(-pi/15, Y)
    f_head = f_head.rotate(pi/15, Z)

    # neck
    f_neck = rounded_cone(0.7, 0.6, 1.2).translate((0, 1.4, -0.9)).rotate(pi/9, X)

    f_chest = (
        sphere(1.3) &
        slab(z0=0, x0=-1, x1=1) &
        plane(-Y).translate((0, 0.6, 0))
    ).translate((0, 1.9, -1.5)).k(0.1)

    f_chest |= f_neck.k(0.2)

    return f_head | f_chest.k(0.2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--samples', default='2^16',
                        help='SDF render samples count')
    parser.add_argument('--seed', type=int,
                        default=np.random.randint(0, 2**32-1),
                        help='random seed')
    parser.add_argument("-o", metavar="KEY=VALUE", nargs='+')
    args = parser.parse_args()

    OVERWRITES = dict((s.split('=')[0], float(s.split("=")[1])) for s in args.o or [])
    np.random.seed(args.seed)

    f = busto()
    #f = grid([busto() for i in range(4)])

    print(f"RANDOM SEED {colored(args.seed, 'green')}")
    print(f"SAMPLES {colored(args.samples, 'green')}")
    save_stl(
        f,
        samples=eval(args.samples.replace("^", "**")),
        name_prefix="busto",
        open_after_saving=True
    )
