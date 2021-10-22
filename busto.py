from sdf import *
from utils import *

def head():
    f = rounded_cone(1, 1, 1)

    return f.scale(1)

def eye():
    f = ellipsoid((2, 1, 1))
    f = f - ellipsoid((1, 0.5, 0.5)).translate((0, -0.8, 0)).k(0.2)
    f = f | sphere(0.5).translate((0, -0.3, 0))

    return f.scale(0.15)

def eyes(pd=0.5):
    eye_1 = eye().translate((-pd/2, 0, 0))
    eye_2 = eye().translate((pd/2, 0, 0))

    return eye_1 | eye_2.k(0.1)

def eye_cavities():
    eyec_1 = sphere(0.4).translate((-0.4, -1, 0.4))
    eyec_2 = sphere(0.4).translate((0.4, -1, 0.4))

    return eyec_1 | eyec_2

def nose():
    f = rounded_cone(0.2, 0.15, 0.8)

    return f.rotate(-pi/10, X)

head = head()

eyec_1 = sphere(0.4).translate((-0.4, -1, 0.4))
eyec_2 = sphere(0.4).translate((0.4, -1, 0.4))

head -= eye_cavities().k(0.2)

eyes = eyes(1).translate((0, -0.8, 0.4))

nose = nose().translate((0, -1, 0))

mouth = ellipsoid((0.75, 0.2, 0.2)).translate((0, -1, -0.55))

f = head | nose.k(0.2) | eyes.k(0.2)
f -= mouth.k(0.2)

save_stl(f, samples=2**20, name_prefix="busto", open_after_saving=True)