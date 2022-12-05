from dataclasses import dataclass
import numpy as np 

@dataclass
class VehicleParameters:
    length: float = 0.17  # length of the car (meters)
    axis_front: float = 0.047  # distance cog and front axis (meters)
    axis_rear: float = 0.05  # distance cog and rear axis (meters)
    front: float = 0.08  # distance cog and front (meters)
    rear: float = 0.08  # distance cog and rear (meters)
    width: float = 0.08  # width of the car (meters)
    height: float = 0.055  # height of the car (meters)
    mass: float = 0.1735  # mass of the car (kg)
    inertia: float = 18.3e-5  # moment of inertia around vertical (kg*m^2)
    
    # input limits
    max_steer: float = 0.384  # max steering angle (radians)
    max_drive: float = 1.0  # maximum drive
    min_drive: float = -1.  # Max drive in reverse 

    # state limits
    min_pos_x: float = -3.  # Minimum position x (m) 
    max_pos_x: float = 3.  # Maximum position x (m)
    min_pos_y: float = -2.
    max_pos_y: float = 2.
    min_vel: float = -0.5 
    max_vel: float =  0.5
    max_heading: float =  2*np.pi   # Maximum heading angle 
    min_heading: float = -2*np.pi   # Minimum heading angle


    """Pacejka 'Magic Formula' parameters.
    Used for magic formula: `peak * sin(shape * arctan(stiffness * alpha))`
    as in Pacejka (2005) 'Tyre and Vehicle Dynamics', p. 161, Eq. (4.6)
    """
    # front
    bf: float = 3.1355  # front stiffness factor
    cf: float = 2.1767  # front shape factor
    df: float = 0.4399  # front peak factor

    # rear
    br: float = 2.8919  # rear stiffness factor
    cr: float = 2.4431  # rear shape factor
    dr: float = 0.6236  # rear peak factor

    # kinematic approximation
    friction: float = 1 # friction parameter
    acceleration: float = 2 # maximum acceleration

    # motor parameters
    cm1: float = 0.3697
    cm2: float = 0.001295
    cr1: float = 0.1629
    cr2: float = 0.02133
