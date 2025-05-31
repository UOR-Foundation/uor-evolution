"""Basic framework for universal intelligence experimentation.

This lightweight package defines simple data structures and an engine for
processing intelligence signals.  It is intentionally minimal so that other
components in the project can begin exploring universal intelligence concepts
without introducing heavy dependencies.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class IntelligenceSignal:
    """Representation of an incoming intelligence signal."""

    source: str
    payload: Any
    strength: float = 1.0


@dataclass
class IntelligenceResult:
    """Aggregated result produced by the intelligence engine."""

    aggregated_payload: Any
    metadata: Dict[str, Any] = field(default_factory=dict)


class UniversalIntelligenceEngine:
    """Engine for analysing lists of :class:`IntelligenceSignal`."""

    async def analyse(self, signals: List[IntelligenceSignal]) -> IntelligenceResult:
        """Aggregate ``signals`` and compute basic metrics."""

        aggregated = [signal.payload for signal in signals]
        average_strength = (
            sum(signal.strength for signal in signals) / (len(signals) or 1)
        )
        metadata = {
            "sources": [signal.source for signal in signals],
            "average_strength": average_strength,
        }
        return IntelligenceResult(aggregated_payload=aggregated, metadata=metadata)


__all__ = [
    "IntelligenceSignal",
    "IntelligenceResult",
    "UniversalIntelligenceEngine",
]
