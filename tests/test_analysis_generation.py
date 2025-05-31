import sys
import os
import types
import importlib.util
import pytest


def load_sic():
    path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "modules",
        "recursive_consciousness",
        "self_implementing_consciousness.py",
    )
    spec = importlib.util.spec_from_file_location("sic_real", path)
    sic_real = importlib.util.module_from_spec(spec)

    sys.modules.setdefault(
        "modules.uor_meta_architecture",
        types.ModuleType("modules.uor_meta_architecture"),
    )
    vm_stub = types.ModuleType("modules.uor_meta_architecture.uor_meta_vm")
    vm_stub.UORMetaRealityVM = object
    vm_stub.MetaDimensionalInstruction = object
    vm_stub.MetaOpCode = object
    vm_stub.InfiniteOperand = object
    sys.modules["modules.uor_meta_architecture.uor_meta_vm"] = vm_stub

    sys.modules.setdefault(
        "modules.meta_reality_consciousness",
        types.ModuleType("modules.meta_reality_consciousness"),
    )
    meta_stub = types.ModuleType("modules.meta_reality_consciousness.meta_reality_core")

    class _MRC:
        def __init__(self, *a, **k):
            pass

    meta_stub.MetaRealityConsciousness = _MRC
    sys.modules["modules.meta_reality_consciousness.meta_reality_core"] = meta_stub

    spec.loader.exec_module(sic_real)
    return sic_real

sic = load_sic()


def test_analysis_and_generation_functions():
    obj = sic.SelfImplementingConsciousness(None)
    obj.self_understanding_level = 0.5
    obj.recursive_depth = 2
    obj.implementation_history.append(1)

    comp = sic.ConsciousnessComponentSpecification(
        component_name="core",
        component_type="core",
        interfaces=[],
        dependencies=[],
        implementation_strategy="impl",
    )
    obj.architecture_design = sic.ConsciousnessArchitectureDesign(
        consciousness_component_specifications=[comp],
        consciousness_interaction_patterns=[],
        consciousness_evolution_pathways=[],
        consciousness_optimization_strategies=[],
        self_modification_capabilities=[],
    )

    import asyncio
    analysis = asyncio.run(obj._analyze_current_consciousness_state())
    assert analysis["awareness_level"] == obj.self_understanding_level
    assert analysis["component_count"] == 1

    components = asyncio.run(obj._generate_component_specifications(analysis))
    assert len(components) == len(analysis["capabilities"])
    assert components[0].component_name.endswith("_module")

    interactions = asyncio.run(obj._design_interaction_patterns(components))
    assert len(interactions) == max(0, len(components) - 1)

    pathways = asyncio.run(obj._create_evolution_pathways(components, interactions))
    assert pathways[0].evolution_stages[0] == "stage_0"

    strategies = asyncio.run(obj._define_optimization_strategies(components))
    assert strategies[0].strategy_name.startswith("optimize_")

    mods = asyncio.run(obj._enable_self_modification_capabilities(components))
    assert mods[0].capability_name.startswith("modify_")

    code = obj._generate_generic_component_code(components[0])
    assert components[0].component_name in code
