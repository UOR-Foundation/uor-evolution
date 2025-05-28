import unittest
from phase1_vm_enhancements import (
    chunk_push, chunk_print, chunk_halt,
    chunk_add, chunk_sub, chunk_mul,
    chunk_input,
    vm_execute,
    chunk_jump
)
from generate_goal_seeker_uor import (
    modify_arithmetic_operands,
    modify_control_flow_target
)
from enhanced_vm_interface import EnhancedVMInterface

class DummyTeacher:
    def __init__(self):
        self.last_guess = None

    def provide_feedback(self, state, last_output=None):
        if last_output is not None:
            self.last_guess = str(last_output)
        else:
            self.last_guess = state.get('output_this_step')
        if self.last_guess is not None and int(self.last_guess) == 42:
            return 1
        return 0

class TestVMInstructions(unittest.TestCase):
    def run_program(self, program):
        output = []
        final_state = {}
        for step in vm_execute(program):
            final_state = step
            if step.get('output_this_step') is not None:
                output.append(step['output_this_step'])
            if step.get('halt_flag') or step.get('error_msg'):
                break
        return ''.join(str(o) for o in output), final_state

    def test_sub(self):
        prog = [chunk_push(5), chunk_push(3), chunk_sub(), chunk_print(), chunk_halt()]
        out, state = self.run_program(prog)
        self.assertEqual(out, str(5 - 3))
        self.assertTrue(state.get('halt_flag'))

    def test_mul(self):
        prog = [chunk_push(4), chunk_push(3), chunk_mul(), chunk_print(), chunk_halt()]
        out, state = self.run_program(prog)
        self.assertEqual(out, str(4 * 3))
        self.assertTrue(state.get('halt_flag'))

    def test_modify_add_operand(self):
        prog = [chunk_push(1), chunk_push(2), chunk_add(), chunk_print(), chunk_halt()]
        modify_arithmetic_operands(prog, [0, 1], 5)
        out, _ = self.run_program(prog)
        self.assertEqual(out, str(5 + 5))

    def test_modify_sub_operand(self):
        prog = [chunk_push(5), chunk_push(2), chunk_sub(), chunk_print(), chunk_halt()]
        modify_arithmetic_operands(prog, [0, 1], 7)
        out, _ = self.run_program(prog)
        self.assertEqual(out, str(7 - 7))

    def test_modify_jump_target(self):
        prog = [
            chunk_push(1),  # placeholder
            chunk_jump(),
            chunk_push(42),
            chunk_print(),
            chunk_push(7),
            chunk_print(),
            chunk_halt(),
        ]
        modify_control_flow_target(prog, 0, 1, 4)
        out, _ = self.run_program(prog)
        self.assertEqual(out, '7')

class TestEnhancedInterface(unittest.TestCase):
    def test_bidirectional(self):
        prog = [chunk_push(42), chunk_print(), chunk_input(), chunk_halt()]
        teacher = DummyTeacher()
        iface = EnhancedVMInterface(prog, teacher=teacher)
        iface.run_until_halt()
        self.assertEqual(teacher.last_guess, '42')
        self.assertTrue(any(e.get('needs_input') for e in iface.events))

if __name__ == '__main__':
    unittest.main()
