import time
import json
import logging
from typing import List, Optional, Iterator, Dict, Any

from phase1_vm_enhancements import vm_execute

logger = logging.getLogger(__name__)

class EnhancedVMInterface:
    """Bridge between Teacher and VM with real-time monitoring."""

    def __init__(self, program: List[int], initial_stack: Optional[List[int]] = None, teacher=None, log_file: str = "vm_events.log"):
        self.program = list(program)
        self.initial_stack = list(initial_stack) if initial_stack else []
        self.teacher = teacher
        self.generator: Optional[Iterator[Dict[str, Any]]] = None
        self.log_file = log_file
        self._setup_logger()
        self.events: List[Dict[str, Any]] = []

    def _setup_logger(self) -> None:
        logging.basicConfig(level=logging.INFO, format="%(message)s")
        self.log_handler = logging.FileHandler(self.log_file)
        self.log_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(self.log_handler)

    def start(self) -> None:
        self.generator = vm_execute(self.program, self.initial_stack)

    def step(self, input_value: Optional[int] = None) -> Dict[str, Any]:
        if self.generator is None:
            self.start()
        try:
            if input_value is not None:
                result = self.generator.send(input_value)
            else:
                result = next(self.generator)
            self._record_event(result)
            if result.get("needs_input") and self.teacher:
                last_output = None
                if len(self.events) >= 2:
                    last_output = self.events[-2].get("output_this_step")
                feedback = self.teacher.provide_feedback(result, last_output=last_output)
                return self.step(feedback)
            return result
        except StopIteration:
            return {"halt_flag": True}

    def _record_event(self, state: Dict[str, Any]) -> None:
        event = {"timestamp": time.time(), **state}
        logger.info(json.dumps(event))
        self.events.append(event)

    def run_until_halt(self) -> List[Dict[str, Any]]:
        state = {}
        while True:
            state = self.step()
            if state.get("halt_flag") or state.get("error_msg"):
                break
        return self.events
