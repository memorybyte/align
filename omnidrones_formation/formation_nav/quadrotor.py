"""
Pure-PyTorch rigid-body multirotor model mirroring OmniDrones' actuator and force model.

Used to validate controller gains and as a higher-fidelity CPU surrogate
(`FormationNavLite(dynamics="quadrotor")`) without Isaac Sim.

What is replicated from OmniDrones (omni_drones/actuators/rotor_group.py,
omni_drones/robots/drone/multirotor.py):
  * rotor command in [-1, 1] -> target throttle sqrt((cmd + 1) / 2); throttle follows it with a
    per-physics-step blend of 0.43; thrust = throttle^2 * max_thrust, yaw moment
    = throttle^2 * max_moment * (-direction);
  * thrust acts along each rotor's z axis at its position (arm_lengths, rotor_angles), yaw
    moments act about the body z axis; aerodynamic drag is zero (as in OmniDrones by default);
  * OmniDrones' inter-drone downwash model (`MultirotorBase.downwash`, applied whenever an
    environment has more than one drone): drone i is pushed along drone j's thrust by
        v_ij * T_j,  v_ij = exp(-0.5 (kr r / z)^2) / (1 + kz z)^2,  kr = 2, kz = 0.3,
    z = distance of i below j along j's thrust axis (>= 0), r = lateral distance. It does not
    scale with the drone size: 1 m below a hovering drone the push is 59 % of that drone's
    weight, for a Crazyflie as for a Hummingbird.
Not modelled: PhysX contact details (a simple ground plane is used), rotor-link inertia.

`OMNIDRONES_ASSETS` holds what the simulated USD assets actually contain where it differs from
the parameter yaml (read from the USD files at the pinned commit): the Crazyflie asset
(cf2x_pybullet.usd) weighs 0.0274 kg in total (base 0.027 + 4 x 0.0001 rotor links, yaml:
0.028) and its rotors sit 0.0396 m from the centre (yaml: 0.043). Pass them to
`QuadrotorModel` to reproduce the Isaac Sim model; the controller keeps using the yaml.
"""

from typing import Optional, Sequence

import torch

from .controller import quat_rotate

GRAVITY = 9.81

# As simulated by OmniDrones (commit 9ce7c20): total mass of all rigid bodies and rotor
# distance from the centre, read from the USD assets.
OMNIDRONES_ASSETS = {
    "crazyflie": dict(mass=0.0274, arm_length=0.0396),  # cf2x_pybullet.usd
    "hummingbird": dict(mass=0.716, arm_length=0.17),  # hummingbird.usd (same as the yaml)
}


def downwash_force(pos: torch.Tensor, thrust_w: torch.Tensor, kr: float = 2.0, kz: float = 0.3) -> torch.Tensor:
    """
    OmniDrones' downwash model (omni_drones/robots/drone/multirotor.py, `MultirotorBase.downwash`
    as called from `apply_action` with kz=0.3).

    pos, thrust_w: (..., n, 3) positions and world-frame thrust vectors of the n drones of an
    environment. Returns (..., n, 3): the force on each drone from the drones above it.
    """
    rel = pos.unsqueeze(-3) - pos.unsqueeze(-2)  # rel[..., i, j, :] = p_j - p_i
    d = thrust_w / thrust_w.norm(dim=-1, keepdim=True).clamp_min(1e-9)  # thrust axis of j
    z = (rel * d.unsqueeze(-3)).sum(-1, keepdim=True)  # i below j along j's axis
    r = (rel - z * d.unsqueeze(-3)).norm(dim=-1, keepdim=True)
    z = z.clamp_min(0.0)
    v = torch.exp(-0.5 * torch.square(kr * r / z)) / (1.0 + kz * z) ** 2
    n = pos.shape[-2]
    eye = torch.eye(n, dtype=torch.bool, device=pos.device).unsqueeze(-1)
    v = torch.where(eye, torch.zeros_like(v), torch.nan_to_num(v, nan=0.0))
    return -(v * thrust_w.unsqueeze(-3)).sum(-2)


def _quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        dim=-1,
    )


class QuadrotorModel:
    """
    Batched multirotor. All state tensors have shape (*batch, ...):
      pos (3, world), rot (4, quaternion w x y z), vel (3, world), omega (3, body rates),
      throttle (num_rotors).
    """

    def __init__(
        self,
        uav_params: dict,
        batch: Sequence[int],
        dt: float,
        device="cpu",
        ground_height: Optional[float] = 0.0,
        mass: Optional[float] = None,
        arm_length: Optional[float] = None,
        downwash: bool = False,
        downwash_scale: float = 1.0,
    ):
        """
        mass / arm_length: override the yaml values (e.g. with `OMNIDRONES_ASSETS[name]`).
        downwash: apply OmniDrones' downwash between the drones of the last batch dimension.
        """
        self.dt = float(dt)
        self.device = torch.device(device)
        self.batch = tuple(batch)
        self.ground_height = ground_height
        self.downwash = downwash
        self.downwash_scale = float(downwash_scale)
        rc = uav_params["rotor_configuration"]
        f32 = dict(dtype=torch.float32, device=self.device)
        self.mass = float(uav_params["mass"] if mass is None else mass)
        inertia = uav_params["inertia"]
        self.inertia = torch.tensor([inertia["xx"], inertia["yy"], inertia["zz"]], **f32)
        max_rot = torch.as_tensor(rc["max_rotation_velocities"], **f32)
        self.max_thrust = max_rot.square() * torch.as_tensor(rc["force_constants"], **f32)
        self.max_moment = max_rot.square() * torch.as_tensor(rc["moment_constants"], **f32)
        self.directions = torch.as_tensor(rc["directions"], **f32)
        angles = torch.as_tensor(rc["rotor_angles"], **f32)
        arms = torch.as_tensor(rc["arm_lengths"], **f32)
        if arm_length is not None:
            arms = torch.full_like(arms, float(arm_length))
        self.roll_arm = torch.sin(angles) * arms  # torque about x per unit thrust
        self.pitch_arm = -torch.cos(angles) * arms  # torque about y per unit thrust
        self.num_rotors = int(rc["num_rotors"])
        self.tau_up = 0.43
        self.tau_down = 0.43

        self.pos = torch.zeros(*self.batch, 3, **f32)
        self.rot = torch.zeros(*self.batch, 4, **f32)
        self.rot[..., 0] = 1.0
        self.vel = torch.zeros(*self.batch, 3, **f32)
        self.omega = torch.zeros(*self.batch, 3, **f32)
        self.throttle = torch.zeros(*self.batch, self.num_rotors, **f32)
        # external force applied in the last step (world frame), like OmniDrones' `drone.forces`
        self.external_force = torch.zeros(*self.batch, 3, **f32)

    # ------------------------------------------------------------------------------------
    @property
    def hover_throttle_cmd(self) -> float:
        """Rotor command that holds the weight at steady state."""
        return float(2.0 * self.mass * GRAVITY / self.max_thrust.sum() - 1.0)

    @property
    def thrust_to_weight(self) -> float:
        return float(self.max_thrust.sum() / (self.mass * GRAVITY))

    def reset(self, idx, pos: torch.Tensor, rot: Optional[torch.Tensor] = None):
        self.pos[idx] = pos
        if rot is None:
            self.rot[idx] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device)
        else:
            self.rot[idx] = rot
        self.vel[idx] = 0.0
        self.omega[idx] = 0.0
        self.throttle[idx] = 0.0
        self.external_force[idx] = 0.0

    def root_state(self) -> torch.Tensor:
        """(*batch, 13): position, quaternion, linear velocity (world), angular velocity (world)."""
        return torch.cat([self.pos, self.rot, self.vel, quat_rotate(self.rot, self.omega)], dim=-1)

    def axes(self):
        """heading (body x) and up (body z) vectors in the world frame."""
        ex = torch.zeros_like(self.pos)
        ex[..., 0] = 1.0
        ez = torch.zeros_like(self.pos)
        ez[..., 2] = 1.0
        return quat_rotate(self.rot, ex), quat_rotate(self.rot, ez)

    # ------------------------------------------------------------------------------------
    def step(self, cmds: torch.Tensor):
        """Advance one physics step with rotor commands (*batch, num_rotors) in [-1, 1]."""
        target = torch.sqrt(torch.clamp((cmds + 1.0) / 2.0, 0.0, 1.0))
        tau = torch.where(target > self.throttle, self.tau_up, self.tau_down)
        self.throttle = self.throttle + tau * (target - self.throttle)
        t = torch.clamp(self.throttle.square(), 0.0, 1.0)
        f = t * self.max_thrust
        yaw_moment = (t * self.max_moment * -self.directions).sum(-1)

        thrust = f.sum(-1, keepdim=True)
        torque = torch.stack([(f * self.roll_arm).sum(-1), (f * self.pitch_arm).sum(-1), yaw_moment], dim=-1)

        force_body = torch.cat([torch.zeros_like(thrust), torch.zeros_like(thrust), thrust], dim=-1)
        thrust_w = quat_rotate(self.rot, force_body)
        if self.downwash and self.pos.dim() >= 2 and self.pos.shape[-2] > 1:
            self.external_force = downwash_force(self.pos, thrust_w) * self.downwash_scale
        else:
            self.external_force = torch.zeros_like(thrust_w)
        acc = (thrust_w + self.external_force) / self.mass
        acc[..., 2] -= GRAVITY
        I = self.inertia
        omega_dot = (torque - torch.cross(self.omega, self.omega * I, dim=-1)) / I

        dt = self.dt
        self.vel = self.vel + acc * dt
        self.pos = self.pos + self.vel * dt
        self.omega = self.omega + omega_dot * dt
        # quaternion integration with the body rates
        angle = self.omega.norm(dim=-1, keepdim=True) * dt
        axis = self.omega / self.omega.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        dq = torch.cat([torch.cos(angle / 2), axis * torch.sin(angle / 2)], dim=-1)
        self.rot = _quat_mul(self.rot, dq)
        self.rot = self.rot / self.rot.norm(dim=-1, keepdim=True)

        if self.ground_height is not None:
            # resting on the ground: no penetration; while thrust cannot lift the drone it stays put
            below = self.pos[..., 2] <= self.ground_height
            if below.any():
                grounded = below & (thrust[..., 0] + self.external_force[..., 2] < self.mass * GRAVITY)
                self.pos[..., 2] = torch.where(below, torch.full_like(self.pos[..., 2], self.ground_height), self.pos[..., 2])
                self.vel[..., 2] = torch.where(below, self.vel[..., 2].clamp_min(0.0), self.vel[..., 2])
                g3 = grounded.unsqueeze(-1)
                self.vel = torch.where(g3, torch.zeros_like(self.vel), self.vel)
                self.omega = torch.where(g3, torch.zeros_like(self.omega), self.omega)
                level = torch.zeros_like(self.rot)
                level[..., 0] = 1.0
                self.rot = torch.where(g3, level, self.rot)
