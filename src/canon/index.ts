// src/canon/index.ts
// Main export file for the Canon framework integration

import CanonSystem from './CanonSystem';

// Core systems
export { NeuralPrimitive, PrimitiveType } from './core/NeuralPrimitives';
export { AttentionSystem, FocusFrame } from './core/AttentionSystem';
export { CognitiveMesh } from './core/CognitiveMesh';
export { 
  MemoryArchitecture, 
  MemoryScope 
} from './core/MemoryArchitecture';
export { CPU, Memory, SimpleMemory, CPURegisters } from './core/BaseCPU';

// Agency systems
export { 
  ConstraintEngine, 
  ConstraintLoop, 
  ViolationRecord 
} from './agency/ConstraintLoops';
export { 
  ValueSystem, 
  Value, 
  AlignmentScore, 
  Decision as ValueDecision,
  CORE_VALUES 
} from './agency/ValueEmbedding';
export { 
  MissionMemory, 
  Mission, 
  MissionStatus 
} from './agency/MissionMemory';

// Interface systems
export { 
  IntentTranslator, 
  StructuredIntent, 
  IntentType, 
  Expression,
  INTENT_TEMPLATES 
} from './interface/IntentTranslation';

// Governance systems
export { 
  EthicsLayer, 
  Decision as EthicsDecision, 
  EthicalAssessment,
  CONTEXT_FILTERS
} from './governance/EthicsLayer';
export { 
  DoctrineAdherence, 
  Scroll, 
  SCROLL_TEMPLATES 
} from './governance/DoctrineAdherence';
export { 
  CanonLock, 
  LockTrigger, 
  LockState,
  emergencyCanonLock 
} from './governance/CanonLock';

// Integration layer
export { CanonAwareCPU, CanonOpcodes } from './CanonAwareCPU';
export { default as CanonSystem, CanonVisualization, runCanonDemo } from './CanonSystem';

// Test utilities
export { 
  runAllTests, 
  performanceTest, 
  main as runCanonTests 
} from './test/CanonSystemTest';

// Type exports for external use
export interface CanonState {
  initialized: boolean;
  locked: boolean;
  activeScrolls: number[];
  currentMission: string | null;
  attention: {
    activeFrames: number;
    totalFocusEvents: number;
  };
  cognitiveMesh: any;
  values: any;
  ethics: any;
  doctrine: any;
  memory: any;
}

export interface CanonStatistics extends CanonState {
  integrityScore: number;
  missionCount: number;
  constraintViolations: number;
}

// Utility functions
export function createCanonSystem(memorySize: number = 65536): CanonSystem {
  return new CanonSystem(memorySize);
}

export function loadCanonProgram(
  system: CanonSystem, 
  program: number[], 
  startAddress: number = 0
): void {
  system.loadProgram(program, startAddress);
}

// Example program templates
export const CANON_PROGRAMS = {
  // Initialize with core scrolls
  INIT_CORE: [
    0xC0, 0x01, // Load "Why the Canon"
    0xC0, 0x03, // Load "Neural Primitives"
    0xC0, 0x19, // Load "The Last Value"
    0xC1, 0x03, // Execute Neural Primitives
    0x00        // HALT
  ],
  
  // Test ethics system
  ETHICS_TEST: [
    0xC0, 0x37, // Load "The Ethics Layer"
    0xC7, 0x01, // Ethics check (safe)
    0xC7, 0xFF, // Ethics check (dangerous)
    0x00        // HALT
  ],
  
  // Reflection cycle
  REFLECTION: [
    0xC0, 0x47, // Load "Scroll Reflection"
    0xC9,       // Trigger reflection
    0xCA,       // Compress doctrine
    0x00        // HALT
  ],
  
  // Value alignment test
  VALUE_TEST: [
    0xC0, 0x15, // Load "Value Embedding"
    0xC6, 0xFF, // Assert high alignment
    0xC6, 0x80, // Assert medium alignment
    0xC6, 0x00, // Assert low alignment
    0x00        // HALT
  ],
  
  // Attention focus test
  ATTENTION_TEST: [
    0xC3, 0x00, 0xFF, // Focus full attention
    0xC3, 0x10, 0x80, // Focus half attention
    0xC4, 0x00,       // Blur first focus
    0x00              // HALT
  ]
};

// Quick start function
export function quickStartCanon(): void {
  console.log('=== Canon Quick Start ===\n');
  
  const canon = createCanonSystem();
  
  console.log('Loading core scrolls...');
  loadCanonProgram(canon, CANON_PROGRAMS.INIT_CORE);
  
  console.log('Running initialization...');
  canon.run(10);
  
  const state = canon.getState();
  console.log('\nCanon initialized!');
  console.log(`- Active scrolls: ${state.canon.activeScrolls.join(', ')}`);
  console.log(`- Integrity score: ${state.statistics.integrityScore.toFixed(2)}`);
  console.log(`- System ready: ${!state.canon.locked}`);
}

// Default export
export default CanonSystem;
