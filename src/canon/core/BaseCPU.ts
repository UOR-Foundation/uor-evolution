// src/canon/core/BaseCPU.ts
// Base CPU class for Canon integration

export interface CPURegisters {
  A: number;    // Accumulator
  X: number;    // X register
  Y: number;    // Y register
  PC: number;   // Program Counter
  SP: number;   // Stack Pointer
  FLAGS: number; // Status flags
}

export interface Memory {
  read(address: number): number;
  write(address: number, value: number): void;
}

export class CPU {
  protected registers: CPURegisters;
  protected memory: Memory;
  protected halted: boolean = false;
  
  constructor(memory: Memory) {
    this.memory = memory;
    this.registers = {
      A: 0,
      X: 0,
      Y: 0,
      PC: 0,
      SP: 0xFF,
      FLAGS: 0
    };
  }
  
  executeInstruction(instruction: number): void {
    // Base implementation - to be overridden
    const opcode = instruction & 0xFF;
    
    switch (opcode) {
      case 0x00: // NOP
        break;
      case 0xEA: // NOP (alternative)
        break;
      default:
        console.log(`Unknown opcode: ${opcode.toString(16)}`);
    }
    
    this.registers.PC++;
  }
  
  halt(): void {
    this.halted = true;
  }
  
  reset(): void {
    this.registers.A = 0;
    this.registers.X = 0;
    this.registers.Y = 0;
    this.registers.PC = 0;
    this.registers.SP = 0xFF;
    this.registers.FLAGS = 0;
    this.halted = false;
  }
  
  setFlag(flag: string, value: boolean): void {
    const flagBits: { [key: string]: number } = {
      'C': 0x01, // Carry
      'Z': 0x02, // Zero
      'I': 0x04, // Interrupt disable
      'D': 0x08, // Decimal mode
      'B': 0x10, // Break
      'V': 0x40, // Overflow
      'N': 0x80  // Negative
    };
    
    if (flagBits[flag]) {
      if (value) {
        this.registers.FLAGS |= flagBits[flag];
      } else {
        this.registers.FLAGS &= ~flagBits[flag];
      }
    }
  }
  
  getFlag(flag: string): boolean {
    const flagBits: { [key: string]: number } = {
      'C': 0x01,
      'Z': 0x02,
      'I': 0x04,
      'D': 0x08,
      'B': 0x10,
      'V': 0x40,
      'N': 0x80
    };
    
    return (this.registers.FLAGS & flagBits[flag]) !== 0;
  }
  
  getState() {
    return {
      registers: { ...this.registers },
      halted: this.halted
    };
  }
}

// Simple memory implementation
export class SimpleMemory implements Memory {
  private data: Uint8Array;
  
  constructor(size: number = 65536) {
    this.data = new Uint8Array(size);
  }
  
  read(address: number): number {
    return this.data[address & 0xFFFF];
  }
  
  write(address: number, value: number): void {
    this.data[address & 0xFFFF] = value & 0xFF;
  }
  
  clear(): void {
    this.data.fill(0);
  }
}
