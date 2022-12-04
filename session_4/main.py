import matplotlib.patches as patch
import matplotlib.pyplot as plt
import numpy as np
from rcracers.simulator import simulate
from rcracers.simulator.dynamics import KinematicBicycle
from plotting import *
from animation import AnimateParking
from parameters import VehicleParameters
from typing import Callable, Tuple
import sys
import casadi as cs
import os
WORKING_DIR = os.path.split(__file__)[0]
sys.path.append(os.path.join(WORKING_DIR, os.pardir))


plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.size": 14
})

# w x h of the parking spot. Just for visualization purposes.
PARK_DIMS = np.array((0.25, 0.12))

π = np.pi


def x2T(x: np.array) -> np.ndarray:
    φ = x[2]

    T = np.zeros((3, 3), dtype=np.float64)

    T[:2, :2] = np.array([[np.cos(φ), -np.sin(φ)],
                          [np.cos(φ),  np.sin(φ)]])
    T[0, 2] = x[0]
    T[1, 2] = x[1]

    return T


def create_cover_circles(l: np.float64, w: np.float64, n_c: int):
    d = l / (2*n_c)
    r = np.sqrt(d**2 + (w**2) / 4)

    center = []

    for k in range(n_c):
        center.append(np.array([(2*k + 1) * d - l/2, 0, 1]))  # Homogeneous

    return center, r


def plot_cover_circle(x: np.float64, l: np.float64, w: np.float64, center: list[np.ndarray], radius: np.float64) -> None:
    ax = plt.gca()

    for c in center:
        circ = patch.Circle(c, radius, fill=True,
                            linewidth=1.5, color="#445491", fc="#738ADE", alpha=0.5)
        ax.add_patch(circ)

    vehicle_rect = patch.Rectangle(
        [x[0]-l/2, x[1]-w/2], l, w, angle=np.rad2deg(x[2]), rotation_point="center",
        fill=False, linewidth=2.5, color="#DB8276")
    ax.add_patch(vehicle_rect)
    ax.set_aspect("equal")

    plt.title(r"Covering circle: with vehicle pose")
    plt.xlabel(r"$x$")
    plt.xlim(-2, 6)
    plt.ylabel(r"$y$")
    plt.ylim(-1, 5)
    plt.show()

def main():
    l = 4
    w = 2
    n_c = 3
    center, radius = create_cover_circles(l, w, n_c)

    x = np.array([2, 2, π/4, 0])
    T = x2T(x)

    center_T = []
    for c in center:
        center_T.append(T@c)

    plot_cover_circle(x, l, w, center_T, radius)


if __name__ == "__main__":
    main()
