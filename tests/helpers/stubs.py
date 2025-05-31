"""Test helper stubs for missing core classes.

These lightweight placeholders satisfy imports without
relying on heavy or unfinished implementations.
"""

class PrimeInstruction:
    """Minimal stand in for :class:`core.instruction_set.PrimeInstruction`."""
    pass

class StateTransitionManager:
    """Simplified state transition manager for tests."""
    def get_possible_transitions(self, state):
        return []

    def execute_transition(self, transition):
        return None
