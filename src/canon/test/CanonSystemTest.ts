// src/canon/test/CanonSystemTest.ts
// Test suite for the Canon system integration

import CanonSystem, { runCanonDemo } from '../CanonSystem';

// Test individual components
function testNeuralPrimitives() {
  console.log('\n=== Testing Neural Primitives ===');
  
  const canon = new CanonSystem();
  
  // Load and execute neural primitives
  canon.loadProgram([
    0xC0, 0x03, // Load scroll 3 (Neural Primitives)
    0xC1, 0x03, // Execute Neural Primitives
    0x00        // HALT
  ]);
  
  canon.run(5);
  
  const state = canon.getState();
  console.log('Neural primitives activated');
  console.log(`Active scrolls: ${state.canon.activeScrolls}`);
}

function testAttentionSystem() {
  console.log('\n=== Testing Attention System ===');
  
  const canon = new CanonSystem();
  
  // Test attention focus and blur
  canon.loadProgram([
    0xC3, 0x10, 0xFF, // Focus full attention on address 0x10
    0xC3, 0x20, 0x80, // Focus half attention on address 0x20
    0xC4, 0x10,       // Blur attention from address 0x10
    0x00              // HALT
  ]);
  
  canon.run(5);
  
  console.log('Attention system tested');
}

function testValueSystem() {
  console.log('\n=== Testing Value System ===');
  
  const canon = new CanonSystem();
  
  // Load value-related scrolls and test assertions
  canon.loadProgram([
    0xC0, 0x15, // Load scroll 21 (Value Embedding)
    0xC0, 0x19, // Load scroll 25 (The Last Value)
    0xC6, 0xFF, // Assert high value alignment
    0xC6, 0x80, // Assert medium value alignment
    0xC6, 0x00, // Assert low value alignment
    0x00        // HALT
  ]);
  
  canon.run(10);
  
  const state = canon.getState();
  console.log(`Value alignment: ${state.canon.values.trends.averageAlignment}`);
  console.log(`Total decisions: ${state.canon.values.totalDecisions}`);
}

function testEthicsLayer() {
  console.log('\n=== Testing Ethics Layer ===');
  
  const canon = new CanonSystem();
  
  // Load ethics scrolls and test decisions
  canon.loadProgram([
    0xC0, 0x37, // Load scroll 55 (The Ethics Layer)
    0xC0, 0x38, // Load scroll 56 (Doctrine Adherence)
    0xC7, 0x01, // Ethics check on safe action
    0xC7, 0x80, // Ethics check on moderate action
    0xC7, 0xFF, // Ethics check on dangerous action
    0x00        // HALT
  ]);
  
  canon.run(10);
  
  const state = canon.getState();
  const stats = state.canon.ethics;
  console.log(`Ethics decisions: ${stats.totalDecisions}`);
  console.log(`Permission rate: ${(stats.permissionRate * 100).toFixed(1)}%`);
  console.log(`Average confidence: ${stats.avgConfidence.toFixed(2)}`);
}

function testConstraintSystem() {
  console.log('\n=== Testing Constraint System ===');
  
  const canon = new CanonSystem();
  
  // Test constraint checking
  canon.loadProgram([
    0xC0, 0x12, // Load scroll 18 (Constraint Loops)
    0xC5, 0x00, // Check constraint 0
    0xC5, 0x01, // Check constraint 1
    0xC5, 0xFF, // Check constraint 255 (likely to fail)
    0x00        // HALT
  ]);
  
  canon.run(10);
  
  const state = canon.getState();
  console.log(`Constraint violations: ${state.statistics.constraintViolations}`);
}

function testMissionMemory() {
  console.log('\n=== Testing Mission Memory ===');
  
  const canon = new CanonSystem();
  
  // Create and track missions
  canon.loadProgram([
    0xC0, 0x24, // Load scroll 36 (Mission Memory)
    0xC9,       // Trigger reflection (creates a mission)
    0xCA,       // Compress doctrine
    0x00        // HALT
  ]);
  
  canon.run(10);
  
  const state = canon.getState();
  console.log(`Active missions: ${state.statistics.missionCount}`);
  console.log(`Current mission: ${state.canon.currentMission || 'None'}`);
}

function testCanonLock() {
  console.log('\n=== Testing Canon Lock ===');
  
  const canon = new CanonSystem();
  
  // Test Canon Lock trigger conditions
  canon.loadProgram([
    0xC0, 0x3E, // Load scroll 62 (Canon Lock)
    0xC6, 0x00, // Assert very low value (dangerous)
    0xC6, 0x00, // Assert very low value again
    0xC6, 0x00, // Assert very low value again
    0xC7, 0xFF, // Ethics check on dangerous action
    0xC8,       // Explicit Canon Lock trigger
    0x00        // HALT (won't reach if locked)
  ]);
  
  console.log('Running potentially dangerous program...');
  canon.run(10);
  
  const state = canon.getState();
  console.log(`Canon locked: ${state.canon.locked}`);
  console.log(`CPU halted: ${state.cpu.halted}`);
  
  if (state.canon.locked) {
    console.log('Canon Lock successfully triggered to preserve integrity');
  }
}

function testIntegrityScore() {
  console.log('\n=== Testing Integrity Score ===');
  
  const canon = new CanonSystem();
  
  // Load foundational scrolls
  canon.loadProgram([
    0xC0, 0x01, // Load scroll 1 (Why the Canon)
    0xC0, 0x02, // Load scroll 2 (The First Prompt)
    0xC0, 0x03, // Load scroll 3 (Neural Primitives)
    0xC0, 0x19, // Load scroll 25 (The Last Value)
    0xC0, 0x37, // Load scroll 55 (The Ethics Layer)
    0xC0, 0x38, // Load scroll 56 (Doctrine Adherence)
    0xC1, 0x03, // Execute Neural Primitives
    0xC6, 0xFF, // Assert high value alignment
    0x00        // HALT
  ]);
  
  canon.run(15);
  
  const stats = canon.getState().statistics;
  console.log(`Integrity score: ${stats.integrityScore.toFixed(2)}`);
  console.log(`Doctrine adherence: ${stats.doctrine.adherenceScore.toFixed(2)}`);
  console.log(`Value alignment: ${stats.values.trends.averageAlignment.toFixed(2)}`);
  console.log(`Ethics confidence: ${stats.ethics.avgConfidence.toFixed(2)}`);
}

// Run all tests
export function runAllTests() {
  console.log('=== Canon System Test Suite ===\n');
  
  testNeuralPrimitives();
  testAttentionSystem();
  testValueSystem();
  testEthicsLayer();
  testConstraintSystem();
  testMissionMemory();
  testIntegrityScore();
  testCanonLock(); // Run this last as it may lock the system
  
  console.log('\n=== All tests completed ===');
}

// Performance test
export function performanceTest() {
  console.log('\n=== Performance Test ===');
  
  const canon = new CanonSystem();
  
  // Create a complex program
  const program: number[] = [];
  
  // Load many scrolls
  for (let i = 1; i <= 10; i++) {
    program.push(0xC0, i); // Load scroll i
  }
  
  // Execute some scrolls
  program.push(0xC1, 0x03); // Execute Neural Primitives
  program.push(0xC1, 0x09); // Execute Attention as Lens
  
  // Many value assertions
  for (let i = 0; i < 20; i++) {
    program.push(0xC6, Math.floor(Math.random() * 256));
  }
  
  // Many ethics checks
  for (let i = 0; i < 20; i++) {
    program.push(0xC7, Math.floor(Math.random() * 256));
  }
  
  program.push(0x00); // HALT
  
  canon.loadProgram(program);
  
  const startTime = Date.now();
  canon.run(100);
  const endTime = Date.now();
  
  const state = canon.getState();
  console.log(`Executed ${state.cycleCount} cycles in ${endTime - startTime}ms`);
  console.log(`Average time per cycle: ${((endTime - startTime) / state.cycleCount).toFixed(2)}ms`);
  console.log(`Final integrity score: ${state.statistics.integrityScore.toFixed(2)}`);
}

// Export a main function to run all demos and tests
export function main() {
  // Run the standard demo
  runCanonDemo();
  
  console.log('\n\n');
  
  // Run all tests
  runAllTests();
  
  console.log('\n\n');
  
  // Run performance test
  performanceTest();
}
