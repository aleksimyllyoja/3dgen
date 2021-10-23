from sdf import *
from utils import *

from copy import copy
from termcolor import colored

import argparse

def head():
    jaw = rounded_cone(0.55, 0.75, 0.7).translate((0, -0.3, -0.8))
    jaw = jaw.rotate(-pi/5, X)
    jaw = jaw & sphere(1.3).translate((0, 0, -0.3)).k(0.2)
    #f = f - box((3, 3, 0.5), (0, 0, -0.75)).k(0.2)
    jaw = jaw - box((3, 0.5, 3), (0, -1.2, 0)).k(0.2)

    chin = sphere(0.1).translate((0, -0.9, -0.8))

    return jaw | sphere(0.8).translate((0, -0.3, 0.3)).k(0.2) | chin.k(0.3)

def eye(opening = pi/9):
    f = sphere() & slab(z0=0)
    f2 = copy(f).rotate(-opening, X).translate((0, -0.2, 0))
    f = f.rotate(pi, X).rotate(opening, X)
    f = f | f2 
    f = f | sphere(0.8)
    f = f & plane(-Y)

    return f.scale(0.15)

def eyes(pd=0.5):
    eye_1 = eye().translate((-pd/2, 0, 0))
    eye_2 = eye().translate((pd/2, 0, 0))

    return eye_1 | eye_2.k(0.1)

def eye_cavities(d=0.4, pd=0.7):
    eyec_1 = sphere(d).translate((-pd/2, 0, 0))
    eyec_2 = sphere(d).translate((pd/2, 0, 0))

    return (eyec_1 | eyec_2)

def nose():
    f = rounded_cone(0.2, 0.15, 0.6)

    return f.rotate(-pi/10, X) & plane(Z).k(0.2)

def neck():
    return rounded_cone(0.3, 0.4, 0.8)

def chest():
    return sphere() & slab(z0=0, x0=-0.8, x1=0.8) & plane(-Y).translate((0, 0.3, 0))

def mouth():
    return ellipsoid((0.3, 0.2, 0.1))

def forehead(pd=0.4):
    f = torus(0.3, 0.1).rotate(pi/2, Y).rotate(pi/2, Z)
    f = f.translate((-pd, 0, 0)) | copy(f).translate((pd, 0, 0))

    f = f & sphere(0.7).k(0.2)
    return f.rotate(pi/8, X) & slab(z0=-0.5)

def sdf_path(l, brush_size=0.1, k=0.2):
    blob = sphere(brush_size, l[0])

    for p0, p1 in zip(l, l[1:]):
        blob = blob | sphere(brush_size, p1).k(k)

    return blob

def _forehead():
    brow = sdf_path(bspline(np.array([
        (-1.1, 1, -0.3),
        (-1, 0, 0),
        (-0.5, 0, 0.8),
        (-0.2, 0, -0.5),
        (0, 0, 0),
    ]), n=100))

    brow2 = sdf_path(bspline(np.array([
        (0, 0, 0),
        (0.2, 0, 0),
        (0.5, 0, 0.8),
        (1, 0, 0),
        (1.1, 1, -0.3),
    ]), n=100))

    return (brow | brow2.k(0.1)).rotate(pi/8, X)

def lips():
    upper_lip = sdf_path(bspline(np.array([
        (-1, 0.5, 0),
        (-0.25, 0, 0.2),
        (0, 0, 0),
        (0.25, 0, 0.2),
        (1, 0.5, 0)
    ]), n=50), brush_size=0.1)
    
    lower_lip = sdf_path(bspline(np.array([
        (-1, 0.5, 0),
        (0, 0, -0.5),
        (1, 0.5, 0)
    ]), n=50), brush_size=0.1).translate((0, 0, -0.1))

    upper_lip = upper_lip - ellipsoid((0.2, 0.2, 0.2)).translate((0, 0, 0.4)).k(0.2)

    return upper_lip | lower_lip

def setup():
    parser = argparse.ArgumentParser()
    parser.add_argument('--samples', default='2^16',
                        help='SDF render samples count')
    parser.add_argument('--seed', type=int,
                        default=np.random.randint(0, 2**32-1),
                        help='random seed')
    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = setup()

    print(f"RANDOM SEED {colored(args.seed, 'green')}")

    np.random.seed(args.seed)

    head = head()

    eyes = eyes(0.8).translate((0, -0.84, 0.0))
    eye_cavities = eye_cavities().translate((0, -1.3, 0))

    #f = (head - eye_cavities.k(0.2)) | _forehead().scale(0.6).translate((0, -1, 0.2)).k(0.2)
    f = (head - eye_cavities.k(0.2)) | forehead().translate((0, -0.8, 0.1)).k(0.1)
    f = f | eyes 

    nose = nose().translate((0, -1.2, -0.4))

    neck = neck().translate((0, -0.5, -1))

    chest = chest().translate((0, -0.2, -2))

    mouth = mouth().translate((0, -0.8, -0.6))

    f = f | nose.k(0.2) | neck.k(0.2) | chest.k(0.1)

    lips = lips()

    f = f | lips.scale(0.2).translate((0, -1.1, -0.6)).k(0.1)
    
    print(f"SAMPLES {colored(args.samples, 'green')}")
    save_stl(
        f,
        samples=eval(args.samples.replace("^", "**")),
        name_prefix="busto",
        open_after_saving=True
    )