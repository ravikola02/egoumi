"""Where observation.state / action come from, one class per source.

    none    video-only dataset: no observation.state / action columns (default)
    zeros   14-d all-zero placeholder in the h2r_collection layout, flagged in info.json;
            only for smoke-testing training tooling, never for training
    (planned) gripper   ArUco gripper pose + jaw width (scripts/gripper_pose.py output)

A source returns (state, action) arrays of shape (n, dim), or (None, None) for `none`.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

# Same 14 dims and names as h2r_collection's --eef-point grasp export, so h2r_il tooling reads it unchanged.
H2R_14D_NAMES = [
    "left.eef.x", "left.eef.y", "left.eef.z", "left.eef.roll", "left.eef.pitch", "left.eef.yaw",
    "right.eef.x", "right.eef.y", "right.eef.z", "right.eef.roll", "right.eef.pitch", "right.eef.yaw",
    "left.gripper", "right.gripper",
]
H2R_14D_LAYOUT = "[L pos 3, L rpy 3, R pos 3, R rpy 3, L grip 1, R grip 1]"


def derive_action_from_state(state: np.ndarray) -> np.ndarray:
    """action[t] = state[t+1]; last frame repeats (h2r_collection convention)."""
    return np.concatenate([state[1:], state[-1:]], axis=0)


class NoneSource:
    name = "none"
    dim = 0
    names: List[str] = []
    placeholder = False

    def compute(self, episode, n: int, fps: float) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        return None, None

    def info(self) -> Dict:
        return {"state_source": self.name, "state_placeholder": False,
                "state_signal": "No observation.state / action: video-only dataset."}


class ZerosSource:
    name = "zeros"
    dim = len(H2R_14D_NAMES)
    names = H2R_14D_NAMES
    placeholder = True

    def compute(self, episode, n: int, fps: float):
        state = np.zeros((n, self.dim), np.float32)
        return state, derive_action_from_state(state)

    def info(self) -> Dict:
        return {"state_source": self.name, "state_placeholder": True, "state_layout": H2R_14D_LAYOUT,
                "state_signal": "PLACEHOLDER: all zeros. End-effector pose and gripper values are not "
                                "reconstructed yet. Do not train on this."}


SOURCES = {s.name: s for s in (NoneSource, ZerosSource)}


def get_source(name: str):
    if name not in SOURCES:
        raise ValueError(f"unknown state source {name!r}; choose from {sorted(SOURCES)}")
    return SOURCES[name]()
