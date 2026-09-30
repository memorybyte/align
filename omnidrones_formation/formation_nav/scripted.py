"""
Scripted (non-learning) policy for smoke tests: every drone flies straight to its own formation
slot with a proportional speed command. It ignores neighbours and obstacles, so it checks the
simulation side of the task (take-off, velocity tracking, formation building under downwash,
following the route, holding at the goal) independently of training.
"""

import torch
from tensordict import TensorDictBase

# position of the (clipped) vector to the own slot in the actor observation, see
# FormationNavCore.observations: velocity 3, heading 3, up 3, angular velocity 3, altitude 1,
# phase one-hot 3, then the slot vector
TO_SLOT = slice(16, 19)


class SlotSeeker:
    def __init__(self, max_speed: float, action_mode: str = "dir_speed", gain: float = 1.0):
        self.max_speed = float(max_speed)
        self.action_mode = action_mode
        self.gain = float(gain)

    def actions(self, obs: torch.Tensor) -> torch.Tensor:
        to_slot = obs[..., TO_SLOT]
        dist = to_slot.norm(dim=-1, keepdim=True)
        speed = (self.gain * dist).clamp(max=self.max_speed)
        direction = to_slot / dist.clamp_min(1e-6)
        if self.action_mode == "dir_speed":
            return torch.cat([direction, speed / self.max_speed], dim=-1)
        return direction * speed / self.max_speed

    def __call__(self, td: TensorDictBase) -> TensorDictBase:
        td.set(("agents", "action"), self.actions(td.get(("agents", "observation"))))
        return td
