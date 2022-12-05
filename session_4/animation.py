from typing import Iterable, Tuple
from rcracers.rendering.core import Window, Actor
from rcracers.rendering.scenes import TrajectoryBundel, DEFAULT_LINE_WIDTH
from uuid import uuid4
from rcracers.rendering.objects import Trajectory, ColoredSprite
import numpy as np 
import pyglet


class ParkingSpot:

    def __init__(self, color, position, w, h):
        self._rectangle = pyglet.shapes.BorderedRectangle(
                *position, w, h, 
                border=5*DEFAULT_LINE_WIDTH,
                color=color, border_color=(255,255,255))
        self.position = position

    def draw(self):
        self._rectangle.draw()

class AnimateParking(Window):
    def setup(self, states: np.ndarray, time_step: float = 0.05, *, 
                parking_spot_dims: tuple = (0.25, 0.12),
                obstacle_positions: Iterable = None 
        ):
        self.set_size(720, 520)
        self.camera.center = np.mean(states, axis=0)[:2]
        self.camera.magnify *= 2.
        self.background_color = (190, 190, 190, 1)

        self.time_step = time_step

        # Draw the parking spot as a rectangle.
        park_spot_dims = np.array(parking_spot_dims)
        for i in range(-2, 3):
            position = -0.5*park_spot_dims.copy()
            position[0] += i * park_spot_dims[0]
            if i == 0:
                color = (118,181,197)
            else: 
                color = self.background_color[:3]
            parking_target = ParkingSpot(color, position, *park_spot_dims)
            self.register(f"parkingspot-{i}", parking_target)


        self.vehicle = ColoredSprite("car", position=(0.2, 0), color=(9,103,155))
        self.actor = Actor(self.vehicle, states[:, :3], time_step=time_step, loop=True)
        self.register("actor", self.actor)
        
        if obstacle_positions is None: 
            obstacle_positions = [] 
        for i, obstacle in enumerate(obstacle_positions):
            obstacle_sprite = ColoredSprite("car", color=(226,135,67), position=obstacle)
            self.register(f"obstacle-{i}", obstacle_sprite)

        self.__refs = 0 

    def add_car_trajectory(self, states: np.ndarray, *, color: Tuple[int]): 
        self.__refs += 1
        reference_car = ColoredSprite("car", position=(0.2, 0), color=color)
        actor = Actor(reference_car, states[:, :3], time_step=self.time_step, loop=True)
        self.register(f"actor-{self.__refs}", actor)

    def trace(self, states: np.ndarray, *, width=3, color=(0, 0, 150)):
        if not states.ndim == 2 or states.shape[-1] < 2:
            raise ValueError(
                f"Invalid states array shape. Expected (horizon x nb states) and got {states.shape}"
            )
        name = f"trace-{uuid4()}"
        trace = Trajectory(np.empty((0, 2)), color=color, width=width)
        self.register(name, Actor(trace, states[:, :2]))
        return name

    def bundle(self, states: np.ndarray, *, width=3, color=(0, 0, 150)):
        if not states.ndim == 3 or states.shape[-1] < 2:
            raise ValueError(
                f"Invalid states array shape. Expected (time steps x horizon x nb states) and got {states.shape}"
            )
        name = f"bundle-{uuid4()}"
        bundle = TrajectoryBundel(states[0, :, :2], width=width, color=color)
        self.register(name, Actor(bundle, states[..., :2]))
        return name

