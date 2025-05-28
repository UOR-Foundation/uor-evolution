import time
from phase1_vm_enhancements import chunk_push, chunk_add, chunk_halt, vm_execute
from enhanced_vm_interface import EnhancedVMInterface

PROGRAM = [chunk_push(10), chunk_push(20), chunk_add(), chunk_halt()]
ITERATIONS = 1000

# Baseline direct execution
start = time.time()
for _ in range(ITERATIONS):
    for _ in vm_execute(PROGRAM):
        pass
baseline = time.time() - start

# Via interface
iface = EnhancedVMInterface(PROGRAM)
start = time.time()
for _ in range(ITERATIONS):
    iface.start()
    iface.run_until_halt()
interface_time = time.time() - start

print(f"Baseline time: {baseline:.4f}s")
print(f"Interface time: {interface_time:.4f}s")
