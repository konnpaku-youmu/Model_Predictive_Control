import matplotlib.pyplot as plt
from parameters import VehicleParameters
from matplotlib.patches import Rectangle
import numpy as np


def plot_input_sequence(u_sequence, params: VehicleParameters): 
    """Plot the component of the inputs together with their bounds. 

    Args:
        u_sequence (_type_): Inputs (n_time_steps x input_dim)
        params (VehicleParameters): Parameters to obtain the bounds from 
    """
    plt.subplot(2,2, (1,3))
    plt.title("Control actions")
    plt.plot(u_sequence[:,0], u_sequence[:,1], marker=".") 
    bounds = Rectangle(np.array((params.min_drive, -params.max_steer)), params.max_drive-params.min_drive, 2*params.max_steer, fill=False)
    plt.gca().add_patch(bounds)
    plt.xlabel("$a$")
    plt.ylabel("$\delta$");
    plt.subplot(2,2,2)
    plt.title("Steering angle")
    plt.plot(u_sequence[:,1].squeeze(), marker=".") 
    style=dict(linestyle="--", color="black") 
    plt.axhline(params.max_steer, **style)
    plt.axhline(-params.max_steer, **style)
    plt.ylabel("$\delta$");
    plt.subplot(2,2,4)
    plt.title("Acceleration")
    plt.plot(u_sequence[:,0].squeeze(), marker=".") 
    plt.axhline(params.min_drive, **style)
    plt.axhline(-params.max_drive, **style)
    plt.ylabel("$a$");
    plt.xlabel("$t$")
    plt.tight_layout()


def plot_state_trajectory(x_sequence, title: str = "Trajectory", ax = None, color="tab:blue", label: str="", park_dims: np.ndarray = None):
    """Plot the trajectory of the vehicle as represented by `x_sequence`. 

    Args:
        x_sequence (np.ndarray): Sequence of states shape: (n_steps x state_dim)
        title (str, optional): Title of the plot. Defaults to "Trajectory".
        ax (_type_, optional): Axis to draw on. Defaults to None.
        color (str, optional): Color to use for the vehicle. Defaults to "tab:blue".
        label (str, optional): Label for the legend. Defaults to "".
        park_dims (np.ndarray, optional): dimensions of the parking space. If they are omitted, the parking space is not shown. Defaults to None.
    """
    plt.tight_layout()
    if ax is None:
        ax = plt.gca()
    ax.set_title(title)
    car_params = VehicleParameters()
    if park_dims is not None:
        parking_area = Rectangle(-0.5*park_dims, *park_dims, ec="tab:green", fill=False)
        ax.add_patch(parking_area)
    extra_arg = dict() 
    for i, xt in enumerate(x_sequence):
        if i==len(x_sequence)-1:
            extra_arg["label"]= label
        if i%2 == 0:  # Only plot a subset 
            alpha = min(0.1 + i / len(x_sequence), 1)
            anchor = xt[:2] - 0.5 * np.array([car_params.length, car_params.width])
            car = Rectangle(anchor, car_params.length, car_params.width, 
                angle=xt[2]/np.pi*180.,
                rotation_point="center",
                alpha=alpha, 
                ec="black",
                fc=color,
                **extra_arg
            )
            ax.add_patch(car)
    
    plt.legend()
    plt.grid(True)
    ax.plot(x_sequence[:,0], x_sequence[:,1], marker=".", color="black")
    ax.set_xlabel("$p_x$ [m]")
    ax.set_ylabel("$p_y$ [m]")
    ax.set_aspect("equal")

def plot_states_separately(x_sequence):
    plt.subplot(4,1,1)
    plt.title("Position x")
    plt.plot(x_sequence[:,0].squeeze(), marker=".") 
    plt.ylabel("$p_x$");
    plt.subplot(4,1,2)
    plt.title("Position y")
    plt.plot(x_sequence[:,1].squeeze(), marker=".") 
    plt.ylabel("$y$")
    plt.subplot(4,1,3)
    plt.title("Angle")
    plt.plot(x_sequence[:,2].squeeze(), marker=".")
    plt.ylabel("$\psi$")
    plt.subplot(4,1,4)
    plt.title("Velocity")
    plt.plot(x_sequence[:,3].squeeze(), marker=".") 
    plt.ylabel("$v$")
    plt.xlabel("$t$")
    plt.tight_layout()
    