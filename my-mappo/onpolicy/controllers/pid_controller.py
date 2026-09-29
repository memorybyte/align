"""
PID Controller for MA-LSTM-PPO.

This module implements a PID controller to convert desired velocity commands
(v_des) to motor commands (RPMs) for drones.

From paper:
- Policy outputs v_des (desired linear velocity + throttle)
- Use PID controller to convert v_des -> motor commands if needed

NOTE: When using gym-pybullet-drones with ActionType.VEL, the internal 
DSLPIDControl is used automatically, so this controller may not be needed.
This implementation is provided for:
1. Environments that require motor-level commands (ActionType.RPM)
2. Custom velocity control requirements
3. Reference implementation
"""

import numpy as np
from typing import Tuple, Optional


class PIDController:
    """
    PID controller for velocity tracking.
    
    Converts desired velocity v_des to control outputs (thrust/RPM).
    """
    
    def __init__(
        self,
        kp: np.ndarray = None,
        ki: np.ndarray = None,
        kd: np.ndarray = None,
        dt: float = 1/30,
        output_limits: Tuple[float, float] = (-1.0, 1.0),
        integrator_limits: Tuple[float, float] = (-10.0, 10.0),
    ):
        """
        Initialize PID controller.
        
        Args:
            kp: Proportional gains (3,) for [x, y, z] - default tuned for CF2X
            ki: Integral gains (3,)
            kd: Derivative gains (3,)
            dt: Control timestep
            output_limits: Output saturation limits
            integrator_limits: Anti-windup limits for integrator
        """
        # Default gains (tuned for Crazyflie-like dynamics)
        self.kp = kp if kp is not None else np.array([0.4, 0.4, 1.25])
        self.ki = ki if ki is not None else np.array([0.05, 0.05, 0.05])
        self.kd = kd if kd is not None else np.array([0.2, 0.2, 0.5])
        
        self.dt = dt
        self.output_limits = output_limits
        self.integrator_limits = integrator_limits
        
        # Internal state
        self.reset()
    
    def reset(self):
        """Reset controller state."""
        self.integral = np.zeros(3)
        self.prev_error = np.zeros(3)
        self.prev_velocity = None
    
    def compute(
        self,
        v_des: np.ndarray,
        v_current: np.ndarray,
    ) -> np.ndarray:
        """
        Compute control output from velocity error.
        
        Args:
            v_des: Desired velocity (3,) [vx, vy, vz]
            v_current: Current velocity (3,) [vx, vy, vz]
            
        Returns:
            control: Control output (3,) - acceleration command
        """
        # Velocity error
        error = v_des - v_current
        
        # Proportional term
        p_term = self.kp * error
        
        # Integral term with anti-windup
        self.integral += error * self.dt
        self.integral = np.clip(
            self.integral, 
            self.integrator_limits[0], 
            self.integrator_limits[1]
        )
        i_term = self.ki * self.integral
        
        # Derivative term (on error)
        d_term = self.kd * (error - self.prev_error) / self.dt
        self.prev_error = error.copy()
        
        # Combine terms
        control = p_term + i_term + d_term
        
        # Saturate output
        control = np.clip(control, self.output_limits[0], self.output_limits[1])
        
        return control


class VelocityToMotorController:
    """
    Controller to convert desired velocity (v_des) to motor commands.
    
    This is a simplified model for when gym-pybullet-drones is used with
    ActionType.RPM instead of ActionType.VEL.
    
    """
    
    def __init__(
        self,
        mass: float = 0.027,  # Crazyflie mass in kg
        gravity: float = 9.81,
        max_thrust: float = 0.6,  # Max thrust per motor
        arm_length: float = 0.0397,
        hover_rpm: float = 14468.429183500699,
        dt: float = 1/30,
    ):
        """
        Initialize velocity to motor controller.
        
        Args:
            mass: Drone mass in kg
            gravity: Gravitational acceleration
            max_thrust: Maximum thrust per motor
            arm_length: Distance from center to motor
            hover_rpm: RPM required for hover
            dt: Control timestep
        """
        self.mass = mass
        self.gravity = gravity
        self.max_thrust = max_thrust
        self.arm_length = arm_length
        self.hover_rpm = hover_rpm
        self.dt = dt
        
        # Velocity PID controller
        self.velocity_pid = PIDController(dt=dt)
        
        # Thrust coefficient (approximate)
        self.kf = 3.16e-10  # Thrust coefficient for CF2X
        
    def reset(self):
        """Reset controller state."""
        self.velocity_pid.reset()
    
    def compute(
        self,
        v_des: np.ndarray,
        current_state: dict,
    ) -> np.ndarray:
        """
        Convert desired velocity to motor RPMs.
        
        Args:
            v_des: Desired velocity (4,) [vx, vy, vz, throttle_scale]
                   where throttle_scale in [-1, 1] modulates hover thrust
            current_state: Dict with 'position', 'velocity', 'orientation', etc.
            
        Returns:
            rpms: Motor RPMs (4,) for each motor
        """
        # Extract current velocity
        v_current = current_state.get('velocity', np.zeros(3))
        
        # Get velocity command (first 3 components)
        v_cmd = v_des[:3]
        throttle_scale = v_des[3] if len(v_des) > 3 else 0.0
        
        # Compute acceleration command from velocity PID
        accel_cmd = self.velocity_pid.compute(v_cmd, v_current)
        
        # Convert to thrust (simplified model)
        # Thrust = mass * (g + az) for z-axis
        thrust_z = self.mass * (self.gravity + accel_cmd[2] + throttle_scale * 2.0)
        
        # Approximate roll/pitch from desired xy acceleration
        # This is a simplified model - real implementation would use full dynamics
        roll_cmd = np.arctan2(accel_cmd[1], self.gravity)
        pitch_cmd = np.arctan2(-accel_cmd[0], self.gravity)
        
        # Clamp angles
        max_angle = 0.5  # radians
        roll_cmd = np.clip(roll_cmd, -max_angle, max_angle)
        pitch_cmd = np.clip(pitch_cmd, -max_angle, max_angle)
        
        # Convert to motor RPMs using mixer matrix (X configuration)
        # Simplified: all motors get base thrust, modified by roll/pitch
        base_rpm = np.sqrt(thrust_z / (4 * self.kf))
        base_rpm = np.clip(base_rpm, 0, 1.5 * self.hover_rpm)
        
        # Mixer for X configuration
        # Motor order: [front-left, front-right, back-right, back-left]
        roll_factor = roll_cmd * 0.1 * self.hover_rpm
        pitch_factor = pitch_cmd * 0.1 * self.hover_rpm
        
        rpms = np.array([
            base_rpm - roll_factor - pitch_factor,  # FL
            base_rpm + roll_factor - pitch_factor,  # FR
            base_rpm + roll_factor + pitch_factor,  # BR
            base_rpm - roll_factor + pitch_factor,  # BL
        ])
        
        # Clamp RPMs
        rpms = np.clip(rpms, 0, 2 * self.hover_rpm)
        
        return rpms


def convert_v_des_to_action(
    v_des: np.ndarray,
    action_type: str = 'vel',
    controller: Optional[VelocityToMotorController] = None,
    current_state: Optional[dict] = None,
) -> np.ndarray:
    """
    Convert desired velocity to appropriate action format.
    
    In paper:
    - If gym-pybullet-drones accepts velocity commands, pass v_des directly
    - If motor-level commands needed, use PID to convert
    
    Args:
        v_des: Desired velocity (4,) [vx, vy, vz, throttle_scale] in [-1, 1]
        action_type: 'vel' for velocity commands, 'rpm' for motor commands
        controller: VelocityToMotorController for RPM conversion
        current_state: Current drone state dict (needed for RPM)
        
    Returns:
        action: Action in appropriate format
    """
    if action_type == 'vel':
        # Pass velocity command directly
        # gym-pybullet-drones ActionType.VEL handles PID internally
        return v_des
    
    elif action_type == 'rpm':
        # Convert to motor RPMs
        if controller is None:
            controller = VelocityToMotorController()
        if current_state is None:
            current_state = {'velocity': np.zeros(3)}
        return controller.compute(v_des, current_state)
    
    else:
        raise ValueError(f"Unknown action type: {action_type}")
