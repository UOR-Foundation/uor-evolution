"""Adaptive teaching logic for PrimeOS backend.

This module contains helper classes used by the Flask
application to track virtual machine performance and
provide new numerical goals. Each class is designed for
clarity and ease of testing.
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional, Tuple

from config_loader import load_config

_CONFIG = load_config()

# Difficulty parameters used by ``AdaptiveTeacher``
DIFFICULTY_LEVELS: Dict[str, Dict[str, int]] = _CONFIG.get(
    "difficulty_levels",
    {
        "EASY": {"range_max": 4},
        "MEDIUM": {"range_max": 9},
        "HARD": {"range_max": 14},
    },
)

class PerformanceMonitor:
    """Track success statistics for the virtual machine."""

    def __init__(self) -> None:
        self.total_attempts: int = 0
        self.successes: int = 0
        self.attempts_per_success: List[int] = []
        self.current_attempts: int = 0
        # Track recent outputs and error magnitudes
        self.recent_outputs: List[int] = []
        self.recent_errors: List[int] = []
        # Maintain historical success counts per parameter/operand
        self.param_stats: Dict[int, Dict[str, int]] = {}

    def record_attempt(self, success: bool) -> None:
        """Record a single attempt outcome."""
        self.total_attempts += 1
        self.current_attempts += 1
        if success:
            self.successes += 1
            self.attempts_per_success.append(self.current_attempts)
            self.current_attempts = 0

    def record_attempt_details(self, operand: int, output: int, target: int) -> None:
        """Record an attempt with extra context for learning heuristics."""
        error = abs(target - output)
        self.recent_outputs.append(output)
        self.recent_errors.append(error)
        if len(self.recent_outputs) > 10:
            self.recent_outputs.pop(0)
        if len(self.recent_errors) > 10:
            self.recent_errors.pop(0)

        stats = self.param_stats.setdefault(operand, {"attempts": 0, "successes": 0})
        stats["attempts"] += 1
        success = error == 0
        if success:
            stats["successes"] += 1

        self.record_attempt(success)

    def success_rate_for_parameter(self, operand: int) -> float:
        """Return success ratio for a specific operand value."""
        stats = self.param_stats.get(operand)
        if not stats or stats["attempts"] == 0:
            return 0.0
        return stats["successes"] / stats["attempts"]

    def recent_average_error(self) -> float:
        """Average of recently observed absolute errors."""
        if not self.recent_errors:
            return 0.0
        return sum(self.recent_errors) / len(self.recent_errors)

    def success_rate(self) -> float:
        """Return recent success rate in [0,1]."""
        if self.total_attempts == 0:
            return 0.0
        denom = len(self.attempts_per_success) + (1 if self.current_attempts else 0)
        if denom == 0:
            return 0.0
        return self.successes / denom

    def average_attempts(self) -> float:
        """Average number of attempts per success."""
        if not self.attempts_per_success:
            return 0.0
        return sum(self.attempts_per_success) / len(self.attempts_per_success)

    def trend(self) -> float:
        """Return improvement trend (negative means improving)."""
        if len(self.attempts_per_success) < 3:
            return 0.0
        recent_avg = sum(self.attempts_per_success[-3:]) / 3
        overall = self.average_attempts()
        return overall - recent_avg

class SequenceGenerator:
    """Generate simple numerical sequences."""

    @staticmethod
    def arithmetic(start: int = 0, step: int = 1, length: int = 5) -> List[int]:
        return [start + step * i for i in range(length)]

    @staticmethod
    def geometric(start: int = 1, ratio: int = 2, length: int = 5) -> List[int]:
        seq = [start]
        for _ in range(1, length):
            seq.append(seq[-1] * ratio)
        return seq

    @staticmethod
    def fibonacci(length: int = 5) -> List[int]:
        seq = [0, 1]
        while len(seq) < length:
            seq.append(seq[-1] + seq[-2])
        return seq[:length]

    @staticmethod
    def primes(length: int = 5) -> List[int]:
        seq: List[int] = []
        n = 2
        while len(seq) < length:
            for p in range(2, int(n ** 0.5) + 1):
                if n % p == 0:
                    break
            else:
                seq.append(n)
            n += 1
        return seq

class AdaptiveCurriculum:
    """Maintain stats about VM performance across number ranges."""

    def __init__(self) -> None:
        self.range_stats: Dict[int, Dict[str, int]] = {}
        self.goal_history: List[Tuple[int, str]] = []

    def _range_key(self, value: int) -> int:
        return (value // 5) * 5

    def register_goal(self, target: int, goal_type: str) -> None:
        self.goal_history.append((target, goal_type))

    def record_attempt(self, target: int, success: bool) -> None:
        key = self._range_key(target)
        stats = self.range_stats.setdefault(key, {"attempts": 0, "successes": 0})
        stats["attempts"] += 1
        if success:
            stats["successes"] += 1

    def weakest_range(self) -> Optional[int]:
        weakest: Optional[int] = None
        lowest = 1.1
        for key, data in self.range_stats.items():
            if data["attempts"] == 0:
                continue
            rate = data["successes"] / data["attempts"]
            if rate < lowest:
                lowest = rate
                weakest = key
        return weakest

    def choose_goal_type(self) -> str:
        if self.weakest_range() is not None and random.random() < 0.4:
            return "reinforcement"
        if random.random() < 0.2:
            return "sequence"
        if random.random() > 0.9:
            return "challenge"
        return "standard"

class AdaptiveTeacher:
    """Provide targets and difficulty adjustments for the VM."""

    def __init__(self) -> None:
        self.monitor = PerformanceMonitor()
        self.curriculum = AdaptiveCurriculum()
        self.sequence_gen = SequenceGenerator()
        self.difficulty: str = _CONFIG.get("teacher", {}).get("difficulty", "MEDIUM")
        self.current_goal: Optional[int] = None
        self.goal_type: Optional[str] = None

    def record_attempt(self, target: int, success: bool) -> None:
        """Update statistics after each attempt."""
        self.monitor.record_attempt(success)
        self.curriculum.record_attempt(target, success)
        self._adjust_difficulty()

    def _adjust_difficulty(self) -> None:
        rate = self.monitor.success_rate()
        if rate > 0.8:
            self.difficulty = "HARD"
        elif rate < 0.5:
            self.difficulty = "EASY"
        else:
            self.difficulty = "MEDIUM"

    def next_goal(self) -> Tuple[int, str]:
        """Return the next numeric goal and a hint."""
        self.goal_type = self.curriculum.choose_goal_type()
        if self.goal_type == "sequence":
            seq = self.sequence_gen.fibonacci(5)
            self.current_goal = seq[-1]
            hint = "follow fibonacci pattern"
        elif self.goal_type == "reinforcement":
            rng = self.curriculum.weakest_range() or 0
            self.current_goal = random.randint(rng, rng + 4)
            hint = "practice weak range"
        elif self.goal_type == "challenge":
            self.current_goal = random.randint(0, 20)
            hint = "challenge mode"
        else:
            max_val = DIFFICULTY_LEVELS[self.difficulty]["range_max"]
            self.current_goal = random.randint(0, max_val)
            hint = "standard goal"
        self.curriculum.register_goal(self.current_goal, self.goal_type)
        return self.current_goal, hint

    def suggest_operand(self, candidates: List[int]) -> int:
        """Choose an operand value based on historical success rates."""
        if not candidates:
            raise ValueError("candidates cannot be empty")
        best = candidates[0]
        best_rate = -1.0
        for val in candidates:
            rate = self.monitor.success_rate_for_parameter(val)
            if rate > best_rate:
                best_rate = rate
                best = val
        return best
