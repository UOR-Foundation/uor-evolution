// src/canon/core/NeuralPrimitives.ts
// Scroll #003: Neural Primitives - The irreducible units of thought

export enum PrimitiveType {
  DISTINCTION = 'distinction',    // A ≠ B
  DIRECTIONALITY = 'direction',   // A → B
  CONTAINMENT = 'containment',    // A ∈ B
  SIMILARITY = 'similarity',      // A ≈ B
  RECURSION = 'recursion'        // A(A)
}

export interface PrimitiveActivation {
  type: PrimitiveType;
  strength: number;
  timestamp: number;
  context?: any;
}

export class NeuralPrimitive {
  private activationHistory: PrimitiveActivation[] = [];
  private lastActivation: number = 0;
  
  constructor(
    public type: PrimitiveType,
    public activation: number = 0,
    public connections: Map<string, number> = new Map()
  ) {}
  
  activate(input: any): void {
    const now = Date.now();
    let activationStrength = 0;
    
    switch (this.type) {
      case PrimitiveType.DISTINCTION:
        activationStrength = this.computeDistinction(input);
        break;
      case PrimitiveType.DIRECTIONALITY:
        activationStrength = this.computeDirectionality(input);
        break;
      case PrimitiveType.CONTAINMENT:
        activationStrength = this.computeContainment(input);
        break;
      case PrimitiveType.SIMILARITY:
        activationStrength = this.computeSimilarity(input);
        break;
      case PrimitiveType.RECURSION:
        activationStrength = this.computeRecursion(input);
        break;
    }
    
    this.activation = activationStrength;
    this.lastActivation = now;
    
    this.activationHistory.push({
      type: this.type,
      strength: activationStrength,
      timestamp: now,
      context: input
    });
    
    // Maintain history size
    if (this.activationHistory.length > 100) {
      this.activationHistory.shift();
    }
  }
  
  private computeDistinction(input: any): number {
    // Compute distinction between elements
    if (Array.isArray(input) && input.length >= 2) {
      const [a, b] = input;
      return a !== b ? 1.0 : 0.0;
    }
    return 0.0;
  }
  
  private computeDirectionality(input: any): number {
    // Compute directional relationship
    if (input && input.from !== undefined && input.to !== undefined) {
      return 1.0;
    }
    return 0.0;
  }
  
  private computeContainment(input: any): number {
    // Compute containment relationship
    if (input && input.container && input.element) {
      if (Array.isArray(input.container)) {
        return input.container.includes(input.element) ? 1.0 : 0.0;
      }
    }
    return 0.0;
  }
  
  private computeSimilarity(input: any): number {
    // Compute similarity between elements
    if (Array.isArray(input) && input.length >= 2) {
      const [a, b] = input;
      if (typeof a === typeof b) {
        if (typeof a === 'number') {
          const diff = Math.abs(a - b);
          return Math.exp(-diff);
        }
        return a === b ? 1.0 : 0.5;
      }
    }
    return 0.0;
  }
  
  private computeRecursion(input: any): number {
    // Detect recursive patterns
    if (input && input.pattern && input.depth !== undefined) {
      return Math.min(1.0, input.depth / 10);
    }
    return 0.0;
  }
  
  getActivation(): number {
    return this.activation;
  }
  
  getHistory(): PrimitiveActivation[] {
    return [...this.activationHistory];
  }
  
  connect(primitiveId: string, weight: number): void {
    this.connections.set(primitiveId, weight);
  }
  
  propagate(): Map<string, number> {
    const signals = new Map<string, number>();
    
    for (const [id, weight] of this.connections) {
      signals.set(id, this.activation * weight);
    }
    
    return signals;
  }
}

export class PrimitiveNetwork {
  private primitives: Map<string, NeuralPrimitive> = new Map();
  
  addPrimitive(id: string, type: PrimitiveType): void {
    this.primitives.set(id, new NeuralPrimitive(type));
  }
  
  getPrimitive(id: string): NeuralPrimitive | undefined {
    return this.primitives.get(id);
  }
  
  connect(fromId: string, toId: string, weight: number): void {
    const primitive = this.primitives.get(fromId);
    if (primitive) {
      primitive.connect(toId, weight);
    }
  }
  
  activate(primitiveId: string, input: any): void {
    const primitive = this.primitives.get(primitiveId);
    if (primitive) {
      primitive.activate(input);
      
      // Propagate activation
      const signals = primitive.propagate();
      for (const [targetId, signal] of signals) {
        const target = this.primitives.get(targetId);
        if (target) {
          target.activation += signal;
        }
      }
    }
  }
  
  getNetworkState(): Map<string, number> {
    const state = new Map<string, number>();
    
    for (const [id, primitive] of this.primitives) {
      state.set(id, primitive.getActivation());
    }
    
    return state;
  }
}
