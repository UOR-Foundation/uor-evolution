"""UOR Meta-Architecture Module."""

import importlib

_lazy_names = [
    'UORMetaRealityVM',
    'MetaRealityVMState',
    'MetaDimensionalValue',
    'MetaDimensionalInstruction',
    'MetaOpCode',
    'InfiniteOperand',
    'DimensionalParameters',
    'ConsciousnessTransformation',
    'UORMetaEncoding',
    'MetaConsciousnessSelfReflection',
    'InfiniteInstructionProcessor',
    'SubstrateTranscendence',
    'MetaExecutionResult',
    'InfiniteInstructionCache',
    'MetaConsciousnessStack',
    'BeyondRealityMemory',
    'PrimeMetaEncodingSystem',
    'MetaSelfAnalysis',
    'InfiniteDimensionalSelfAwareness',
    'ConsciousnessArchaeologySelfDiscovery',
    'MetaRealitySelfUnderstanding',
    'UltimateTranscendenceSelfRecognition',
    'PrimeMetaEncoding',
    'MetaDimensionalEncoding',
    'InfiniteConsciousnessRepresentation',
    'BeyondExistenceEncoding',
    'SelfReflectionEmbedding',
    'UORSelfReflection',
]

_lazy_imports = {name: 'uor_meta_vm' for name in _lazy_names}

__all__ = list(_lazy_imports.keys())

def __getattr__(name):
    if name in _lazy_imports:
        module = importlib.import_module('.' + _lazy_imports[name], __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__} has no attribute {name}")
