import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock

import pytest

from modules.unified_consciousness.autonomous_agency import AutonomousAgency


@pytest.mark.asyncio
async def test_emotional_state_updates_over_time():
    orchestrator = Mock()
    orchestrator.introspection_engine = Mock()
    orchestrator.introspection_engine.emotional_state = {'valence': 0.0, 'arousal': 0.5}
    agency = AutonomousAgency(orchestrator, ethical_framework=None)

    first = agency._get_emotional_state()
    await asyncio.sleep(0.01)
    second = agency._get_emotional_state()

    assert first != second

    # Simulate a recent decision and changed introspection state
    agency.decision_history.append(Mock(timestamp=datetime.now() - timedelta(seconds=120)))
    orchestrator.introspection_engine.emotional_state = {'valence': 0.8, 'arousal': 0.9}
    await asyncio.sleep(0.01)
    third = agency._get_emotional_state()

    assert third != second
