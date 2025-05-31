"""Basic tools for modelling and simulating a simple reality.

This module currently provides lightweight dataclass containers that
represent a minimal "reality" along with a basic engine for advancing a
simulation.  These structures act as stubs so that other parts of the
project can begin experimenting with reality synthesis without relying on
heavy dependencies.  Future components are expected to build on top of
these primitives.
"""

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class RealityModel:
    """Representation of the current simulated reality state."""

    objects: Dict[str, Any] = field(default_factory=dict)
    tick: int = 0


@dataclass
class SimulationParameters:
    """Parameters controlling a single simulation step."""

    time_delta: float = 1.0
    randomness: float = 0.0
    user_input: Dict[str, Any] = field(default_factory=dict)


class RealitySynthesisEngine:
    """Utility for creating models and advancing their state."""

    def create_model(self) -> RealityModel:
        """Return a new, empty :class:`RealityModel`."""

        return RealityModel()

    def simulate_step(self, model: RealityModel, params: SimulationParameters) -> RealityModel:
        """Advance ``model`` using ``params`` and return it.

        The default behaviour simply increments the model's tick and merges
        any ``user_input`` into ``model.objects``.  More sophisticated
        reality modelling logic can replace this in the future.
        """

        model.tick += 1
        if params.user_input:
            model.objects.update(params.user_input)
        return model


__all__ = [
    "RealityModel",
    "SimulationParameters",
    "RealitySynthesisEngine",
]
