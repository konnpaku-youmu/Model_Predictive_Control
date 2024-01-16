from given.homework.problem import NLIterate

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import sys 
import os 

def get_working_dir():
    directory = sys.path[0]
    if directory == "":
        return os.getcwd()
    return directory

from typing import List 

def _prepare_output_path(filename: str, ext: str = "mp4"):
    out_path = os.path.join(get_working_dir(), "output", f"{filename}.{ext}")
    print(f"Saving output to {out_path}")
    os.makedirs(os.path.split(out_path)[0], exist_ok=True)
    return out_path


def animate_iterates(its: List[NLIterate], filename: str = "Iterates.mp4"):

    N, nx = its[0].x.shape
    nu = its[0].u.shape[1]

    fig =plt.figure()
    ax = plt.gca()
    lines = [plt.plot(its[0].x[:,i], label=f"$x_{{{i}}}$")[0] for i in range(nx)] 
    lines += [plt.plot(its[0].u[:,i], label=f"$u_{{{i}}}$")[0] for i in range(nu)]
    plt.title("Iterates")
    plt.xlabel("Predicted time step $k$")
    box_props = dict(boxstyle='round', facecolor='white', alpha=0.5)
    it_label = ax.text(0.9, 0.1, "It. 0", transform=ax.transAxes, bbox=box_props)
    plt.legend()
    def draw_frame(frame, *args):
        for i, line in enumerate(lines):
            if i < nx: 
                line.set_ydata(its[frame].x[:,i])
            else: 
                line.set_ydata(its[frame].u[:,i-nx])
        it_label.set_text(f"It. {frame}")
        return *lines, it_label 

    ani = FuncAnimation(fig, draw_frame, blit=True, frames=len(its))
    out_path = _prepare_output_path(filename)
    ani.save(out_path, fps=12)
    # plt.show()
    draw_frame(len(its)-1)
    out_path = _prepare_output_path(filename, "pdf")
    fig.savefig(out_path)
    plt.show()


def animate_positions(its: List[NLIterate], filename: str = "Iterates.mp4"):
    fig = plt.figure()
    line = plt.plot(*(its[0].x[:,:2].T))[0]
    # Mark initial state
    plt.scatter(*(its[0].x[0,:2].T), fc=None)
    plt.annotate("$x_0$", its[0].x[0,:2])
    it_label = plt.text(0,0, "It. 0")
    def draw_frame(frame, *args):
        line.set_xdata(its[frame].x[:,0])
        line.set_ydata(its[frame].x[:,1])
        it_label.set_text(f"It. {frame}")
        return line, it_label
    ani = FuncAnimation(fig, draw_frame, blit=True, frames=len(its))
    out_path = _prepare_output_path(filename)
    ani.save(out_path, fps=12)
    plt.show()
