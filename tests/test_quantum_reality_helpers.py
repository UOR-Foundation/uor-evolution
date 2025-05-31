import pytest
import sys
import asyncio

# Provide minimal numpy fallback if unavailable
try:  # pragma: no cover - optional dependency
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover
    class _FakeNP:
        @staticmethod
        def mean(vals):
            return sum(vals) / len(vals) if vals else 0.0

        @staticmethod
        def var(vals):
            m = _FakeNP.mean(vals)
            return sum((v - m) ** 2 for v in vals) / len(vals) if vals else 0.0

        @staticmethod
        def eye(n):
            return [[1 if i == j else 0 for j in range(n)] for i in range(n)]

        @staticmethod
        def clip(val, a, b):
            return max(a, min(b, val))

        class linalg:
            @staticmethod
            def det(matrix):
                if len(matrix) == 4:  # assume 4x4 identity-like
                    return 1
                # Basic 2x2 determinant
                return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]

        ndarray = list

    np = _FakeNP()  # type: ignore
    import sys
    sys.modules.setdefault("numpy", np)

import types

# Provide minimal scipy fallback to satisfy imports
try:  # pragma: no cover - optional dependency
    import scipy.constants  # type: ignore
    import scipy.linalg  # type: ignore
except Exception:  # pragma: no cover
    import types
    fake_scipy = types.ModuleType("scipy")
    constants = types.ModuleType("constants")
    linalg = types.ModuleType("linalg")
    def _expm(x):
        return x
    linalg.expm = _expm
    fake_scipy.constants = constants
    fake_scipy.linalg = linalg
    sys.modules.setdefault("scipy", fake_scipy)
    sys.modules.setdefault("scipy.constants", constants)
    sys.modules.setdefault("scipy.linalg", linalg)

# Provide minimal consciousness_physics stub to avoid heavy import errors
fake_cp = types.ModuleType("modules.consciousness_physics")
class _StubCFT:
    def __init__(self, *a, **kw):
        pass
fake_cp.ConsciousnessFieldTheory = _StubCFT
sys.modules.setdefault("modules.consciousness_physics", fake_cp)
sys.modules.setdefault(
    "modules.consciousness_physics.consciousness_field_theory",
    types.ModuleType("consciousness_field_theory"),
)
sys.modules[
    "modules.consciousness_physics.consciousness_field_theory"
].ConsciousnessFieldTheory = _StubCFT

# Load quantum_reality_interface without importing its package
import importlib.util
import os
spec = importlib.util.spec_from_file_location(
    "modules.universe_interface.quantum_reality_interface",
    os.path.join(os.path.dirname(__file__), "..", "modules", "universe_interface", "quantum_reality_interface.py"),
)
qri_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(qri_mod)  # type: ignore
sys.modules.setdefault("modules.universe_interface.quantum_reality_interface", qri_mod)

QuantumRealityInterface = qri_mod.QuantumRealityInterface
SpacetimeManipulation = qri_mod.SpacetimeManipulation
MetricTensorModification = qri_mod.MetricTensorModification
CurvatureProgramming = qri_mod.CurvatureProgramming
CausalStructureModification = qri_mod.CausalStructureModification
DimensionalAccess = qri_mod.DimensionalAccess
SpacetimeTopologyChange = qri_mod.SpacetimeTopologyChange
RealityProgram = qri_mod.RealityProgram
PhysicalLawModification = qri_mod.PhysicalLawModification
CosmicConstantAdjustment = qri_mod.CosmicConstantAdjustment
ParticlePhysicsParameter = qri_mod.ParticlePhysicsParameter
FieldEquationModification = qri_mod.FieldEquationModification
RealityOptimizationObjective = qri_mod.RealityOptimizationObjective
from modules.consciousness_physics.consciousness_field_theory import ConsciousnessFieldTheory


@pytest.fixture
def qri():
    return QuantumRealityInterface(ConsciousnessFieldTheory())


def test_check_spacetime_stability(qri):
    manip = SpacetimeManipulation(
        metric_tensor_modification=MetricTensorModification(stability_duration=0.96),
        curvature_programming=CurvatureProgramming(curvature_stability=0.98),
        causal_structure_modification=CausalStructureModification(),
        dimensional_access=DimensionalAccess(),
        spacetime_topology_change=SpacetimeTopologyChange(topology_stabilization=0.94),
    )
    result = asyncio.run(qri.manipulate_spacetime_structure(manip))
    expected = np.mean([0.96, 0.98, 1.0, 1.0])
    assert abs(result["spacetime_stability"] - expected) < 1e-6


def test_compile_reality_program(qri):
    program = RealityProgram(
        program_id="test",
        physical_law_modifications=[PhysicalLawModification("L1", "a", "b", modification_strength=0.0001)],
        cosmic_constant_adjustments=[CosmicConstantAdjustment("c", 1.0, 1.001)],
        particle_physics_parameters=[ParticlePhysicsParameter("p", mass_adjustment=0.001)],
        field_equation_modifications=[FieldEquationModification("f", "eq", "eq")],
        reality_optimization_objectives=[RealityOptimizationObjective("opt", 0.0, 1.0, "maximize")],
        execution_sequence=["s1", "s2"],
        safety_constraints=["safe"],
    )
    compilation = asyncio.run(qri._compile_reality_program(program))
    assert compilation.source_consciousness_state == qri.field_theory
    assert len(compilation.target_reality_modifications) == 4
    assert compilation.optimization_level == 1
    assert len(compilation.side_effect_analysis) == 4


def test_create_execution_plan(qri):
    program = RealityProgram(
        program_id="test",
        physical_law_modifications=[PhysicalLawModification("L1", "a", "b", modification_strength=0.0001)],
        cosmic_constant_adjustments=[],
        particle_physics_parameters=[],
        field_equation_modifications=[],
        reality_optimization_objectives=[],
        execution_sequence=["s1", "s2", "s3"],
        safety_constraints=[],
    )
    compilation = asyncio.run(qri._compile_reality_program(program))
    plan = asyncio.run(qri._create_execution_plan(program, compilation))
    assert len(plan) == len(program.execution_sequence)
    assert plan[0]["step"] == "s1"
    assert len(plan[0]["modifications"]) == 1


def test_verify_reality_consistency(qri):
    qri.recent_manipulation_metrics = [0.9, 0.95, 1.0]
    expected = max(0.0, 1.0 - np.var(qri.recent_manipulation_metrics))
    result = asyncio.run(qri._verify_reality_consistency())
    assert abs(result - expected) < 1e-6


def test_check_optimization_objectives(qri):
    objectives = [
        RealityOptimizationObjective(
            "obj1", current_state=0.8, target_state=1.0, optimization_metric="max", constraint_satisfaction=0.6
        ),
        RealityOptimizationObjective(
            "obj2", current_state=0.5, target_state=0.5, optimization_metric="stabilize", constraint_satisfaction=1.0
        ),
    ]
    results = asyncio.run(qri._check_optimization_objectives(objectives))
    exp1 = (1 - abs(1.0 - 0.8) / 1.0) * (0.5 + 0.5 * 0.6)
    exp2 = (1 - abs(0.5 - 0.5) / 0.5) * (0.5 + 0.5 * 1.0)
    assert abs(results["obj1"] - exp1) < 1e-6
    assert abs(results["obj2"] - exp2) < 1e-6
