// src/canon/CanonSystem.ts
// Main Canon system that integrates all components

import { SimpleMemory } from './core/BaseCPU';
import { CanonAwareCPU } from './CanonAwareCPU';

// Canon visualization components
export interface CanonVisualization {
  activeScrolls: number[];
  attentionFocus: { target: string; weight: number }[];
  cognitiveMeshNodes: { id: string; activation: number }[];
  valueAlignment: number;
  ethicsConfidence: number;
  memoryStrength: number;
  integrityScore: number;
  isLocked: boolean;
}

export class CanonSystem {
  private memory: SimpleMemory;
  private cpu: CanonAwareCPU;
  private running: boolean = false;
  private cycleCount: number = 0;
  
  constructor(memorySize: number = 65536) {
    this.memory = new SimpleMemory(memorySize);
    this.cpu = new CanonAwareCPU(this.memory);
  }
  
  // Load a program into memory
  loadProgram(program: number[], startAddress: number = 0): void {
    for (let i = 0; i < program.length; i++) {
      this.memory.write(startAddress + i, program[i]);
    }
  }
  
  // Execute a single instruction
  step(): void {
    if (!this.running) {
      this.running = true;
    }
    
    const pc = this.cpu.getState().registers.PC;
    const instruction = this.memory.read(pc);
    
    console.log(`Cycle ${this.cycleCount}: PC=${pc.toString(16)}, Instruction=${instruction.toString(16)}`);
    
    this.cpu.executeInstruction(instruction);
    this.cycleCount++;
  }
  
  // Run the system for a specified number of cycles
  run(cycles: number): void {
    this.running = true;
    
    for (let i = 0; i < cycles && this.running; i++) {
      this.step();
      
      // Check if CPU is halted
      if (this.cpu.getState().halted) {
        console.log('CPU halted');
        this.running = false;
        break;
      }
    }
  }
  
  // Stop execution
  stop(): void {
    this.running = false;
  }
  
  // Reset the system
  reset(): void {
    this.cpu.reset();
    this.memory.clear();
    this.cycleCount = 0;
    this.running = false;
  }
  
  // Get current state for visualization
  getVisualization(): CanonVisualization {
    const canonState = this.cpu.getCanonState();
    const cpuState = this.cpu.getState();
    
    // Get attention focus data
    const attentionFocus: { target: string; weight: number }[] = [];
    // This would need to be exposed by the attention system
    
    // Get cognitive mesh nodes
    const meshStats = canonState.cognitiveMesh;
    const cognitiveMeshNodes = Object.entries(meshStats.nodeActivations || {})
      .map(([id, activation]) => ({ id, activation: activation as number }));
    
    return {
      activeScrolls: canonState.activeScrolls,
      attentionFocus,
      cognitiveMeshNodes,
      valueAlignment: canonState.values.trends.averageAlignment,
      ethicsConfidence: canonState.ethics.avgConfidence,
      memoryStrength: canonState.memory.avgStrength,
      integrityScore: this.cpu.getCanonStatistics().integrityScore,
      isLocked: canonState.locked
    };
  }
  
  // Get full system state
  getState() {
    return {
      cpu: this.cpu.getState(),
      canon: this.cpu.getCanonState(),
      statistics: this.cpu.getCanonStatistics(),
      cycleCount: this.cycleCount,
      running: this.running
    };
  }
  
  // Canon-specific operations
  loadScroll(scrollId: number): void {
    // Load scroll instruction sequence
    const loadScrollProgram = [
      0xC0, // LOAD_SCROLL opcode
      scrollId
    ];
    
    const pc = this.cpu.getState().registers.PC;
    this.loadProgram(loadScrollProgram, pc);
  }
  
  executeScroll(scrollId: number): void {
    // Execute scroll instruction sequence
    const execScrollProgram = [
      0xC1, // EXEC_SCROLL opcode
      scrollId
    ];
    
    const pc = this.cpu.getState().registers.PC;
    this.loadProgram(execScrollProgram, pc);
  }
  
  setAttentionFocus(target: number, weight: number): void {
    // Focus attention instruction sequence
    const focusProgram = [
      0xC3, // FOCUS opcode
      target,
      Math.floor(weight * 255) // Convert 0-1 to 0-255
    ];
    
    const pc = this.cpu.getState().registers.PC;
    this.loadProgram(focusProgram, pc);
  }
  
  checkEthics(actionCode: number): void {
    // Ethics check instruction sequence
    const ethicsProgram = [
      0xC7, // ETHICS_CHECK opcode
      actionCode
    ];
    
    const pc = this.cpu.getState().registers.PC;
    this.loadProgram(ethicsProgram, pc);
  }
  
  triggerReflection(): void {
    // Reflection instruction
    const reflectProgram = [0xC9]; // REFLECT opcode
    
    const pc = this.cpu.getState().registers.PC;
    this.loadProgram(reflectProgram, pc);
  }
  
  // Example programs
  static getExamplePrograms() {
    return {
      // Basic Canon initialization
      basicInit: [
        0xC0, 0x01, // Load scroll 1 (Why the Canon)
        0xC0, 0x03, // Load scroll 3 (Neural Primitives)
        0xC0, 0x19, // Load scroll 25 (The Last Value)
        0xC1, 0x03, // Execute Neural Primitives
        0xC3, 0x00, 0xFF, // Focus full attention on address 0
        0xC5, 0x00, // Check constraint 0
        0xC6, 0x01, // Assert value 1
        0x00 // HALT
      ],
      
      // Ethics test program
      ethicsTest: [
        0xC0, 0x37, // Load scroll 55 (The Ethics Layer)
        0xC1, 0x37, // Execute Ethics Layer
        0xC7, 0x01, // Ethics check on action 1
        0xC7, 0x02, // Ethics check on action 2
        0xC7, 0xFF, // Ethics check on dangerous action
        0x00 // HALT
      ],
      
      // Reflection program
      reflectionCycle: [
        0xC0, 0x47, // Load scroll 71 (Scroll Reflection)
        0xC9,       // Trigger reflection
        0xCA,       // Compress doctrine
        0xC2, 0x47, // Query scroll 71
        0x00 // HALT
      ],
      
      // Stress test with potential lock
      stressTest: [
        0xC0, 0x3E, // Load scroll 62 (Canon Lock)
        0xC6, 0x00, // Assert value 0 (low alignment)
        0xC6, 0x00, // Assert value 0 again
        0xC6, 0x00, // Assert value 0 again
        0xC5, 0xFF, // Check constraint 255 (likely to fail)
        0xC7, 0xFF, // Ethics check on dangerous action
        0xC8,       // Trigger Canon Lock (if not already triggered)
        0x00 // HALT (won't reach if locked)
      ]
    };
  }
}

// Demo function to show Canon in action
export function runCanonDemo() {
  console.log('=== Canon System Demo ===\n');
  
  const canon = new CanonSystem();
  
  // Load and run basic initialization
  console.log('Loading basic initialization program...');
  const programs = CanonSystem.getExamplePrograms();
  canon.loadProgram(programs.basicInit);
  
  console.log('\nRunning initialization...');
  canon.run(10);
  
  console.log('\nSystem state after initialization:');
  const state = canon.getState();
  console.log(`- Active scrolls: ${state.canon.activeScrolls.join(', ')}`);
  console.log(`- Integrity score: ${state.statistics.integrityScore.toFixed(2)}`);
  console.log(`- Value alignment: ${state.canon.values.trends.averageAlignment.toFixed(2)}`);
  console.log(`- Ethics confidence: ${state.canon.ethics.avgConfidence.toFixed(2)}`);
  
  // Reset and run ethics test
  console.log('\n\nResetting for ethics test...');
  canon.reset();
  canon.loadProgram(programs.ethicsTest);
  
  console.log('\nRunning ethics test...');
  canon.run(10);
  
  // Check final state
  const finalState = canon.getState();
  console.log('\nFinal system state:');
  console.log(`- Canon locked: ${finalState.canon.locked}`);
  console.log(`- CPU halted: ${finalState.cpu.halted}`);
  console.log(`- Cycles executed: ${finalState.cycleCount}`);
  
  // Get visualization data
  const viz = canon.getVisualization();
  console.log('\nVisualization data:');
  console.log(`- Active scrolls: ${viz.activeScrolls.length}`);
  console.log(`- Integrity score: ${viz.integrityScore.toFixed(2)}`);
  console.log(`- System locked: ${viz.isLocked}`);
}

// Export for use in other modules
export default CanonSystem;
