from __future__ import annotations

from bisect import bisect_left, bisect_right, insort
from dataclasses import dataclass
import math
from typing import Protocol

import numpy as np


class BendersCut(Protocol):
    constant: float
    x_coefficients: dict[tuple[int, int], float]


@dataclass(frozen=True)
class CanonicalCut:
    constant: float
    coefficients: tuple[float, ...]


def canonical_cut(cut: BendersCut, ordered_keys: tuple[tuple[int, int], ...]) -> CanonicalCut:
    constant = float(cut.constant)
    coefficients = tuple(float(cut.x_coefficients.get(key, 0.0)) for key in ordered_keys)
    if not math.isfinite(constant) or not all(math.isfinite(value) for value in coefficients):
        raise ValueError("Benders cut contains a non-finite coefficient")
    return CanonicalCut(
        constant=0.0 if constant == 0.0 else constant,
        coefficients=tuple(0.0 if value == 0.0 else value for value in coefficients),
    )


def cuts_equal(left: CanonicalCut, right: CanonicalCut, tolerance: float) -> bool:
    return (
        abs(left.constant - right.constant) <= tolerance
        and max(
            (abs(a - b) for a, b in zip(left.coefficients, right.coefficients)),
            default=0.0,
        )
        <= tolerance
    )


class ExactCutRegistry:
    """Index accepted cuts without treating merely similar cuts as duplicates."""

    def __init__(self, ordered_keys: tuple[tuple[int, int], ...], tolerance: float) -> None:
        if not math.isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError("Duplicate-cut tolerance must be finite and positive")
        self.ordered_keys = ordered_keys
        self.tolerance = tolerance
        self._cuts: list[CanonicalCut] = []
        self._constant_index: list[tuple[float, int]] = []
        self._unit_coefficients = np.empty((64, len(ordered_keys)), dtype=float)

    def canonicalize(self, cut: BendersCut) -> CanonicalCut:
        return canonical_cut(cut, self.ordered_keys)

    def is_duplicate(self, candidate: CanonicalCut) -> bool:
        lower = bisect_left(
            self._constant_index,
            (candidate.constant - self.tolerance, -1),
        )
        upper = bisect_right(
            self._constant_index,
            (candidate.constant + self.tolerance, len(self._cuts)),
        )
        return any(
            cuts_equal(candidate, self._cuts[index], self.tolerance)
            for _, index in self._constant_index[lower:upper]
        )

    def nearest_cosine_similarity(self, candidate: CanonicalCut) -> float | None:
        if not self._cuts:
            return None
        vector = np.asarray(candidate.coefficients, dtype=float)
        norm = float(np.linalg.norm(vector))
        if norm == 0.0:
            zero_rows = np.linalg.norm(
                self._unit_coefficients[: len(self._cuts)], axis=1
            ) == 0.0
            return 1.0 if bool(np.any(zero_rows)) else 0.0
        similarities = self._unit_coefficients[: len(self._cuts)] @ (vector / norm)
        return float(np.max(similarities))

    def add(self, candidate: CanonicalCut) -> None:
        index = len(self._cuts)
        self._cuts.append(candidate)
        insort(self._constant_index, (candidate.constant, index))
        vector = np.asarray(candidate.coefficients, dtype=float)
        norm = float(np.linalg.norm(vector))
        unit = vector if norm == 0.0 else vector / norm
        if index == self._unit_coefficients.shape[0]:
            expanded = np.empty((2 * index, len(self.ordered_keys)), dtype=float)
            expanded[:index] = self._unit_coefficients
            self._unit_coefficients = expanded
        self._unit_coefficients[index] = unit

    def __len__(self) -> int:
        return len(self._cuts)
