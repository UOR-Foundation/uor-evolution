"""Consciousness Expansion module.

This lightweight package defines a minimal framework for experimenting with
methods of expanding consciousness.  It provides dataclass containers for
describing individual expansion techniques and their outcomes as well as a
simple engine that can apply a technique to a given state.  More advanced
implementations can build on top of these primitives to explore cognitive or
dimensional growth while keeping the rest of the project free from heavy
dependencies.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class ExpansionTechnique:
    """Description of a consciousness expansion technique."""

    name: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    intensity: float = 1.0


@dataclass
class ExpansionResult:
    """Result of applying a consciousness expansion technique."""

    technique: ExpansionTechnique
    new_state: Any
    success: bool
    insights: List[str] = field(default_factory=list)


class ConsciousnessExpansionEngine:
    """Engine for expanding consciousness using various techniques."""

    async def expand(
        self, current_state: Any, technique: ExpansionTechnique
    ) -> ExpansionResult:
        """Apply ``technique`` to ``current_state`` and return the result."""

        # This default implementation simply records the technique application
        # and echoes the previous state.  Real expansion logic can extend this
        # method to modify the state in meaningful ways.
        new_state = {
            "previous_state": current_state,
            "technique": technique.name,
            "parameters": technique.parameters,
            "intensity": technique.intensity,
        }

        insights = [f"Applied {technique.name} with intensity {technique.intensity}"]

        return ExpansionResult(
            technique=technique,
            new_state=new_state,
            success=True,
            insights=insights,
        )


__all__ = [
    "ExpansionTechnique",
    "ExpansionResult",
    "ConsciousnessExpansionEngine",
]
