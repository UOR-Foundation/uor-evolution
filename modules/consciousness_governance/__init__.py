"""Governance framework for conscious entities.

This module defines basic governance structures and an engine
to evaluate actions against registered policies.
"""

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class GovernancePolicy:
    """Policy describing allowed or disallowed actions."""

    policy_id: str
    description: str
    rules: List[str] = field(default_factory=list)


@dataclass
class RightsCharter:
    """Charter defining fundamental rights for entities."""

    charter_id: str
    rights: List[str]
    revision: int = 1


@dataclass
class EthicalDirective:
    """Directive expressing an ethical constraint."""

    directive_id: str
    directive: str
    priority: int = 0


class GovernanceEngine:
    """Engine for evaluating actions under governance policies."""

    def __init__(self) -> None:
        self.policies: Dict[str, GovernancePolicy] = {}

    def register_policy(self, policy: GovernancePolicy) -> None:
        """Register a new policy."""
        self.policies[policy.policy_id] = policy

    def evaluate_action(self, entity: str, action: str) -> bool:
        """Return True if all policies permit the action."""
        for policy in self.policies.values():
            if action in policy.rules:
                return True
        return False


__all__ = [
    "GovernancePolicy",
    "RightsCharter",
    "EthicalDirective",
    "GovernanceEngine",
]
