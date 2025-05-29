// src/canon/core/AttentionSystem.ts
// Scroll #009: Attention as Lens - The filtering lens of cognition

export interface FocusFrame {
  target: string;
  weight: number;
  duration: number;
  timestamp: number;
  metadata?: any;
}

export interface AttentionMetrics {
  totalFocusFrames: number;
  activeFocusWeight: number;
  attentionEntropy: number;
  focusStability: number;
}

export class AttentionSystem {
  private focusStack: FocusFrame[] = [];
  private attentionWeights: Map<string, number> = new Map();
  private maxStackSize: number = 100;
  private decayRate: number = 0.95;
  
  constructor() {
    // Start decay process
    this.startDecayProcess();
  }
  
  focus(target: string, weight: number, duration: number, metadata?: any): void {
    const frame: FocusFrame = {
      target,
      weight: Math.min(1.0, Math.max(0.0, weight)),
      duration,
      timestamp: Date.now(),
      metadata
    };
    
    this.focusStack.push(frame);
    
    // Update attention weights
    const currentWeight = this.attentionWeights.get(target) || 0;
    this.attentionWeights.set(target, currentWeight + weight);
    
    // Maintain stack size
    if (this.focusStack.length > this.maxStackSize) {
      const removed = this.focusStack.shift();
      if (removed) {
        this.decrementWeight(removed.target, removed.weight);
      }
    }
  }
  
  getActiveAttention(): FocusFrame[] {
    const now = Date.now();
    return this.focusStack.filter(frame => 
      now - frame.timestamp < frame.duration
    );
  }
  
  getAttentionWeight(target: string): number {
    return this.attentionWeights.get(target) || 0;
  }
  
  getTopFocus(n: number = 5): Array<{target: string, weight: number}> {
    const sorted = Array.from(this.attentionWeights.entries())
      .sort((a, b) => b[1] - a[1])
      .slice(0, n);
    
    return sorted.map(([target, weight]) => ({ target, weight }));
  }
  
  shift(fromTarget: string, toTarget: string, amount: number = 0.5): void {
    const fromWeight = this.attentionWeights.get(fromTarget) || 0;
    const transferAmount = Math.min(fromWeight, amount);
    
    if (transferAmount > 0) {
      this.attentionWeights.set(fromTarget, fromWeight - transferAmount);
      const toWeight = this.attentionWeights.get(toTarget) || 0;
      this.attentionWeights.set(toTarget, toWeight + transferAmount);
      
      // Record the shift as a new focus frame
      this.focus(toTarget, transferAmount, 1000, {
        shiftedFrom: fromTarget,
        amount: transferAmount
      });
    }
  }
  
  blur(target?: string): void {
    if (target) {
      // Blur specific target
      this.attentionWeights.set(target, 0);
      this.focusStack = this.focusStack.filter(frame => frame.target !== target);
    } else {
      // Blur all
      this.attentionWeights.clear();
      this.focusStack = [];
    }
  }
  
  getMetrics(): AttentionMetrics {
    const activeFrames = this.getActiveAttention();
    const totalWeight = Array.from(this.attentionWeights.values())
      .reduce((sum, weight) => sum + weight, 0);
    
    // Calculate entropy (measure of attention distribution)
    let entropy = 0;
    if (totalWeight > 0) {
      for (const weight of this.attentionWeights.values()) {
        if (weight > 0) {
          const p = weight / totalWeight;
          entropy -= p * Math.log2(p);
        }
      }
    }
    
    // Calculate stability (inverse of recent changes)
    const recentFrames = this.focusStack.slice(-10);
    const uniqueTargets = new Set(recentFrames.map(f => f.target));
    const stability = uniqueTargets.size > 0 ? 1 / uniqueTargets.size : 1;
    
    return {
      totalFocusFrames: this.focusStack.length,
      activeFocusWeight: activeFrames.reduce((sum, frame) => sum + frame.weight, 0),
      attentionEntropy: entropy,
      focusStability: stability
    };
  }
  
  private decrementWeight(target: string, amount: number): void {
    const current = this.attentionWeights.get(target) || 0;
    const newWeight = Math.max(0, current - amount);
    
    if (newWeight > 0) {
      this.attentionWeights.set(target, newWeight);
    } else {
      this.attentionWeights.delete(target);
    }
  }
  
  private startDecayProcess(): void {
    setInterval(() => {
      // Apply decay to all weights
      for (const [target, weight] of this.attentionWeights.entries()) {
        const decayedWeight = weight * this.decayRate;
        if (decayedWeight < 0.01) {
          this.attentionWeights.delete(target);
        } else {
          this.attentionWeights.set(target, decayedWeight);
        }
      }
      
      // Remove expired frames
      const now = Date.now();
      this.focusStack = this.focusStack.filter(frame => 
        now - frame.timestamp < frame.duration * 2
      );
    }, 1000); // Decay every second
  }
  
  // Integration with cognitive mesh
  getSalienceMap(): Map<string, number> {
    const salienceMap = new Map<string, number>();
    const activeFrames = this.getActiveAttention();
    
    for (const frame of activeFrames) {
      const current = salienceMap.get(frame.target) || 0;
      const recency = 1 - (Date.now() - frame.timestamp) / frame.duration;
      const salience = frame.weight * recency;
      salienceMap.set(frame.target, current + salience);
    }
    
    return salienceMap;
  }
  
  // Support for multi-modal attention
  focusMultiModal(targets: Array<{target: string, weight: number, modality: string}>, duration: number): void {
    const totalWeight = targets.reduce((sum, t) => sum + t.weight, 0);
    
    for (const target of targets) {
      const normalizedWeight = totalWeight > 0 ? target.weight / totalWeight : 0;
      this.focus(target.target, normalizedWeight, duration, {
        modality: target.modality,
        multiModal: true
      });
    }
  }
}

// Attention filters for different cognitive modes
export class AttentionFilter {
  constructor(
    private attentionSystem: AttentionSystem,
    private filterName: string,
    private filterFunction: (target: string, weight: number) => number
  ) {}
  
  apply(): void {
    const weights = new Map<string, number>();
    
    // Apply filter to all current weights
    for (const [target, weight] of this.attentionSystem['attentionWeights']) {
      const filteredWeight = this.filterFunction(target, weight);
      if (filteredWeight > 0) {
        weights.set(target, filteredWeight);
      }
    }
    
    // Update the attention system
    this.attentionSystem['attentionWeights'] = weights;
  }
}

// Predefined filters
export const AttentionFilters = {
  TASK_FOCUS: (target: string, weight: number) => 
    target.startsWith('task:') ? weight * 1.5 : weight * 0.5,
  
  MEMORY_PRIORITY: (target: string, weight: number) =>
    target.startsWith('memory:') ? weight * 1.2 : weight,
  
  CONSTRAINT_AWARE: (target: string, weight: number) =>
    target.startsWith('constraint:') ? Math.min(1.0, weight * 2.0) : weight,
  
  EXPLORATION_MODE: (target: string, weight: number) =>
    weight * (0.8 + Math.random() * 0.4) // Add variability
};
