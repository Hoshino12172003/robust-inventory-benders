from __future__ import annotations

from dataclasses import dataclass
import math
import random


@dataclass(frozen=True)
class GapPolicyState:
    iteration: int
    benders_gap: float
    previous_benders_gap: float
    lower_bound: float | None
    upper_bound: float | None

    @property
    def log_gap(self) -> float:
        return math.log(max(self.benders_gap, 1e-12))

    @property
    def gap_improvement(self) -> float:
        return math.log(max(self.previous_benders_gap, 1e-12) / max(self.benders_gap, 1e-12))


class GapPolicy:
    def select_gap(self, state: GapPolicyState) -> float:
        raise NotImplementedError


class ExactGapPolicy(GapPolicy):
    def __init__(self, final_gap: float) -> None:
        self.final_gap = final_gap

    def select_gap(self, state: GapPolicyState) -> float:
        return self.final_gap


class FixedGapPolicy(GapPolicy):
    def __init__(self, gap: float) -> None:
        self.gap = gap

    def select_gap(self, state: GapPolicyState) -> float:
        return self.gap


class RLInspiredGapPolicy(GapPolicy):
    """Discrete action-style MIPGap rule inspired by the RL-iGBD source."""

    def __init__(self, lower: float, upper: float, actions: int = 11) -> None:
        self.lower = lower
        self.upper = upper
        self.actions = actions

    def select_gap(self, state: GapPolicyState) -> float:
        if state.iteration <= 1:
            action_index = self.actions - 1
        elif state.gap_improvement < 0.02:
            action_index = max(0, self.actions // 3)
        elif state.gap_improvement < 0.15:
            action_index = self.actions // 2
        else:
            action_index = min(self.actions - 1, int(self.actions * 0.75))
        return self.gap_from_action(action_index, state.benders_gap)

    def gap_from_action(self, action_index: int, benders_gap: float) -> float:
        action_index = min(max(0, int(action_index)), self.actions - 1)
        raw = -1.0 + 2.0 * action_index / float(self.actions - 1)
        adaptive_upper = min(self.upper, max(self.lower, benders_gap))
        gap = self.lower + (raw + 1.0) * 0.5 * (adaptive_upper - self.lower)
        return float(min(max(gap, self.lower), self.upper))


class RandomGapPolicy(RLInspiredGapPolicy):
    def select_gap(self, state: GapPolicyState) -> float:
        return self.gap_from_action(random.randrange(self.actions), state.benders_gap)
