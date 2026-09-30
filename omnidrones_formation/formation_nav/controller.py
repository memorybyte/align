"""
Velocity-tracking geometric controller (Lee, Leok & McClamroch, 2010) for FormationNav.

The maths is identical to OmniDrones' `LeePositionController`
(omni_drones/controllers/lee_position_controller.py, MIT licence) and a test checks that the
rotor commands match it exactly. The differences:
  * the gains: OmniDrones reads them from `omni_drones/controllers/cfg/lee_controller_<drone>.yaml`
    and ships no such file for the Crazyflie. Here they live in `LEE_GAINS` below (or the task
    config), so nothing inside the OmniDrones installation has to be edited;
  * the mass can be set to the simulated one (the Crazyflie yaml says 0.028 kg, the asset
    weighs 0.0274 kg);
  * optional integral action on the velocity error (with anti-windup), like the velocity PID of
    the Crazyflie firmware. Off by default (then the controller is stateless and matches
    OmniDrones exactly). The feed-forward `target_acc` is used by FormationNav to cancel the
    external force OmniDrones applies (its inter-drone downwash model).

Pure PyTorch: it also drives the quadrotor model in `quadrotor.py` for tests on CPU.
"""

from typing import Dict, Optional, Sequence

import torch
import torch.nn as nn

# Gains per drone model, keyed by the "name" field of the drone's parameter yaml.
# Units follow OmniDrones: position / velocity gains act on accelerations (1/s^2, 1/s);
# attitude / angular-rate gains are torque gains (N m / rad, N m s / rad) that are divided
# by the inertia internally, i.e. the closed-loop angular acceleration is
#   alpha = -(attitude_gain / I) * e_R - (angular_rate_gain / I) * omega.
LEE_GAINS: Dict[str, Dict[str, list]] = {
    # Copied from OmniDrones omni_drones/controllers/cfg/lee_controller_<name>.yaml
    "hummingbird": dict(
        position_gain=[4.0, 4.0, 4.0],
        velocity_gain=[2.2, 2.2, 2.2],
        attitude_gain=[0.7, 0.7, 0.035],
        angular_rate_gain=[0.1, 0.1, 0.025],
    ),
    "firefly": dict(
        position_gain=[6.0, 6.0, 6.0],
        velocity_gain=[4.7, 4.7, 4.7],
        attitude_gain=[3.0, 3.0, 0.15],
        angular_rate_gain=[0.52, 0.52, 0.18],
    ),
    "neo11": dict(
        position_gain=[8.0, 8.0, 17.0],
        velocity_gain=[6.0, 6.0, 10.0],
        attitude_gain=[4.0, 4.0, 2.0],
        angular_rate_gain=[0.7, 0.7, 0.7],
    ),
    # Crazyflie 2.x (OmniDrones has no gains for it): the Hummingbird gains scaled by the
    # ratio of the inertias (Ixx = Iyy = 1.4e-5, Izz = 2.17e-5 vs 0.007 / 0.012 kg m^2), so the
    # closed-loop attitude dynamics are the same well-damped ones:
    #   roll / pitch: alpha = -100 e - 14.3 omega   (omega_n = 10 rad/s, zeta = 0.71)
    #   yaw:          alpha = -2.92 e - 2.08 omega
    # Position / velocity gains act on accelerations, so they carry over unchanged.
    # Validated on the quadrotor model in quadrotor.py (tests/test_controller.py).
    "crazyflie": dict(
        position_gain=[4.0, 4.0, 4.0],
        velocity_gain=[2.2, 2.2, 2.2],
        attitude_gain=[1.4e-3, 1.4e-3, 6.33e-5],
        angular_rate_gain=[2.0e-4, 2.0e-4, 4.52e-5],
    ),
}


# ----------------------------------------------------------------------------------------
# quaternion helpers (w, x, y, z), same formulas as omni_drones.utils.torch
# ----------------------------------------------------------------------------------------


def normalize(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x / (torch.norm(x, dim=-1, keepdim=True) + eps)


def quat_rotate(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate v from the body frame to the world frame."""
    q_w, q_vec = q[..., :1], q[..., 1:]
    a = v * (2.0 * q_w.square() - 1.0)
    b = torch.cross(q_vec, v, dim=-1) * q_w * 2.0
    c = q_vec * (q_vec * v).sum(-1, keepdim=True) * 2.0
    return a + b + c


def quat_rotate_inverse(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate v from the world frame to the body frame."""
    q_w, q_vec = q[..., :1], q[..., 1:]
    a = v * (2.0 * q_w.square() - 1.0)
    b = torch.cross(q_vec, v, dim=-1) * q_w * 2.0
    c = q_vec * (q_vec * v).sum(-1, keepdim=True) * 2.0
    return a - b + c


def quaternion_to_rotation_matrix(q: torch.Tensor) -> torch.Tensor:
    w, x, y, z = torch.unbind(q, dim=-1)
    tx, ty, tz = 2.0 * x, 2.0 * y, 2.0 * z
    twx, twy, twz = tx * w, ty * w, tz * w
    txx, txy, txz = tx * x, ty * x, tz * x
    tyy, tyz, tzz = ty * y, tz * y, tz * z
    m = torch.stack(
        [
            1 - (tyy + tzz), txy - twz, txz + twy,
            txy + twz, 1 - (txx + tzz), tyz - twx,
            txz - twy, tyz + twx, 1 - (txx + tyy),
        ],
        dim=-1,
    )
    return m.unflatten(-1, (3, 3))


def quaternion_to_yaw(q: torch.Tensor) -> torch.Tensor:
    w, x, y, z = torch.unbind(q, dim=-1)
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def mixer_matrix(rotor_config: dict, inertia: dict) -> torch.Tensor:
    """(num_rotors, 4): [angular acceleration (3), total thrust] -> per-rotor thrust."""
    angles = torch.as_tensor(rotor_config["rotor_angles"], dtype=torch.float64)
    arms = torch.as_tensor(rotor_config["arm_lengths"], dtype=torch.float64)
    kf = torch.as_tensor(rotor_config["force_constants"], dtype=torch.float64)
    km = torch.as_tensor(rotor_config["moment_constants"], dtype=torch.float64)
    directions = torch.as_tensor(rotor_config["directions"], dtype=torch.float64)
    A = torch.stack([
        torch.sin(angles) * arms,
        -torch.cos(angles) * arms,
        -directions * km / kf,
        torch.ones_like(angles),
    ])
    I = torch.diag(torch.tensor([inertia["xx"], inertia["yy"], inertia["zz"], 1.0], dtype=torch.float64))
    return (A.T @ torch.linalg.inv(A @ A.T) @ I).float()


# ----------------------------------------------------------------------------------------
# controller
# ----------------------------------------------------------------------------------------


class LeeVelocityController(nn.Module):
    """
    Geometric controller tracking a velocity (and optionally position / acceleration) and a
    yaw angle; outputs rotor commands in [-1, 1] as expected by OmniDrones' rotor model
    (thrust proportional to (cmd + 1) / 2 at steady state).

        cmd = controller.compute(root_state, target_vel=v, target_yaw=psi)

    root_state: (..., 13) = position, quaternion (w, x, y, z), linear velocity (world),
    angular velocity (world).

    Integral action (`integral_gain` > 0 on an axis, needs `dt` = the period between calls):
        I <- clip(I + integral_gain * (target_vel - vel) * dt, -integral_limit, integral_limit)
    is added to the desired acceleration. `compute(..., integrate=mask)` freezes it where mask is
    False (e.g. on the ground) and `reset(idx)` clears it (at episode resets).
    """

    def __init__(
        self,
        uav_params: dict,
        gains: Optional[dict] = None,
        g: float = 9.81,
        mass: Optional[float] = None,
        dt: Optional[float] = None,
        integral_gain: Optional[Sequence[float]] = None,
        integral_limit: Optional[Sequence[float]] = None,
    ):
        super().__init__()
        name = uav_params.get("name", "")
        if gains is None:
            if name not in LEE_GAINS:
                raise KeyError(
                    f"No default controller gains for drone '{name}'. "
                    f"Pass `gains` (task config: controller.gains). Known: {sorted(LEE_GAINS)}"
                )
            gains = LEE_GAINS[name]
        inertia = uav_params["inertia"]
        rotor_config = uav_params["rotor_configuration"]
        I_diag = torch.tensor([inertia["xx"], inertia["yy"], inertia["zz"]])

        self.register_buffer("pos_gain", torch.as_tensor(gains["position_gain"], dtype=torch.float32))
        self.register_buffer("vel_gain", torch.as_tensor(gains["velocity_gain"], dtype=torch.float32))
        self.register_buffer("attitude_gain", torch.as_tensor(gains["attitude_gain"], dtype=torch.float32) / I_diag)
        self.register_buffer("ang_rate_gain", torch.as_tensor(gains["angular_rate_gain"], dtype=torch.float32) / I_diag)
        self.register_buffer("mass", torch.tensor(float(mass if mass is not None else uav_params["mass"])))
        self.register_buffer("g", torch.tensor([0.0, 0.0, abs(g)]))
        max_rot_vel = torch.as_tensor(rotor_config["max_rotation_velocities"], dtype=torch.float32)
        kf = torch.as_tensor(rotor_config["force_constants"], dtype=torch.float32)
        self.register_buffer("max_thrusts", max_rot_vel.square() * kf)
        self.register_buffer("mixer", mixer_matrix(rotor_config, inertia))
        self.num_rotors = int(rotor_config["num_rotors"])

        ki = torch.zeros(3) if integral_gain is None else torch.as_tensor(integral_gain, dtype=torch.float32)
        limit = torch.full((3,), float("inf")) if integral_limit is None else torch.as_tensor(integral_limit, dtype=torch.float32)
        if ki.shape != (3,) or limit.shape != (3,):
            raise ValueError("integral_gain and integral_limit need 3 values (x, y, z)")
        self.use_integral = bool((ki > 0).any())
        if self.use_integral and not dt:
            raise ValueError("integral action needs dt (the time between two compute() calls)")
        self.dt = float(dt) if dt else 0.0
        self.register_buffer("integral_gain", ki)
        self.register_buffer("integral_limit", limit)
        self.register_buffer("integral", torch.zeros(0), persistent=False)

    def reset(self, idx=None):
        """Clear the integral state (all of it, or the batch entries `idx`)."""
        if self.integral.numel() == 0:
            return
        if idx is None:
            self.integral.zero_()
        else:
            self.integral[idx] = 0.0

    def compute(
        self,
        root_state: torch.Tensor,
        target_pos: Optional[torch.Tensor] = None,
        target_vel: Optional[torch.Tensor] = None,
        target_acc: Optional[torch.Tensor] = None,
        target_yaw: Optional[torch.Tensor] = None,
        body_rate: bool = False,
        integrate: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch = root_state.shape[:-1]
        pos, rot, vel, ang_vel = torch.split(root_state, [3, 4, 3, 3], dim=-1)
        target_pos = pos if target_pos is None else target_pos.expand(batch + (3,))
        target_vel = torch.zeros_like(vel) if target_vel is None else target_vel.expand(batch + (3,))
        target_acc = torch.zeros_like(vel) if target_acc is None else target_acc.expand(batch + (3,))
        if target_yaw is None:
            target_yaw = quaternion_to_yaw(rot).unsqueeze(-1)
        else:
            if target_yaw.shape[-1] != 1:
                target_yaw = target_yaw.unsqueeze(-1)
            target_yaw = target_yaw.expand(batch + (1,))
        if not body_rate:
            ang_vel = quat_rotate_inverse(rot, ang_vel)

        # "acc" is minus the desired acceleration (OmniDrones' sign convention)
        acc = (pos - target_pos) * self.pos_gain + (vel - target_vel) * self.vel_gain - self.g - target_acc
        if self.use_integral:
            if self.integral.shape != vel.shape:
                self.integral = torch.zeros_like(vel)
            step = (target_vel - vel) * self.integral_gain * self.dt
            if integrate is not None:
                step = step * integrate.unsqueeze(-1).to(step.dtype)
            self.integral = torch.maximum(torch.minimum(self.integral + step, self.integral_limit), -self.integral_limit)
            acc = acc - self.integral
        R = quaternion_to_rotation_matrix(rot)
        b1_des = torch.cat([torch.cos(target_yaw), torch.sin(target_yaw), torch.zeros_like(target_yaw)], dim=-1)
        b3_des = -normalize(acc)
        b2_des = normalize(torch.cross(b3_des, b1_des, dim=-1))
        R_des = torch.stack([torch.cross(b2_des, b3_des, dim=-1), b2_des, b3_des], dim=-1)
        e = 0.5 * (R_des.transpose(-2, -1) @ R - R.transpose(-2, -1) @ R_des)
        ang_error = torch.stack([e[..., 2, 1], e[..., 0, 2], e[..., 1, 0]], dim=-1)
        ang_acc = -ang_error * self.attitude_gain - ang_vel * self.ang_rate_gain
        thrust = -self.mass * (acc * R[..., :, 2]).sum(-1, keepdim=True)
        rotor_thrust = torch.cat([ang_acc, thrust], dim=-1) @ self.mixer.T
        return rotor_thrust / self.max_thrusts * 2.0 - 1.0

    def forward(self, root_state: torch.Tensor, target_vel: torch.Tensor, target_yaw: torch.Tensor) -> torch.Tensor:
        return self.compute(root_state, target_vel=target_vel, target_yaw=target_yaw)
