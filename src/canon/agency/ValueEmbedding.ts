// src/canon/agency/ValueEmbedding.ts
// Scroll #021: Value Embedding - Structural integration of ethics and preferences

import { ConstraintLoop, ConstraintEngine } from './ConstraintLoops';
import { MemoryArchitecture } from '../core/MemoryArchitecture';

export interface Value {
  id: string;
  weight: number;
  constraints: string[];
  scrollReference: number;
  description: string;
  metadata?: any;
}

export interface Decision {
  id: string;
  action: string;
  context: any;
  timestamp: number;
  valueAlignments: Map<string, number>;
}

export interface AlignmentScore {
  score: number;
  violations: string[];
  recommendation: string;
  valueBreakdown: Map<string, number>;
}

export interface Action {
  type: string;
  target?: string;
  parameters?: any;
  expectedOutcome?: string;
}

export class ValueSystem {
  private values: Map<string, Value> = new Map();
  private decisionHistory: Decision[] = [];
  private constraintEngine: ConstraintEngine;
  private memoryArchitecture?: MemoryArchitecture;
  private maxHistorySize: number = 1000;
  
  constructor(memoryArchitecture?: MemoryArchitecture) {
    this.constraintEngine = new ConstraintEngine();
    this.memoryArchitecture = memoryArchitecture;
  }
  
  embedValue(value: Value): void {
    this.values.set(value.id, value);
    
    // Create constraints for this value
    for (const constraintId of value.constraints) {
      const constraint = new ConstraintLoop(
        `value_${value.id}_${constraintId}`,
        () => this.checkValueConstraint(value.id, constraintId),
        () => this.enforceValueConstraint(value.id, constraintId),
        value.weight * 10, // Priority based on value weight
        { valueId: value.id, constraintId }
      );
      
      this.constraintEngine.addConstraint(constraint);
    }
    
    // Store in memory if available
    if (this.memoryArchitecture) {
      this.memoryArchitecture.store(
        value,
        value.scrollReference,
        'value_system',
        `embedded_value_${value.id}`
      );
    }
  }
  
  removeValue(valueId: string): boolean {
    const value = this.values.get(valueId);
    if (!value) return false;
    
    // Remove associated constraints
    for (const constraintId of value.constraints) {
      this.constraintEngine.removeConstraint(`value_${valueId}_${constraintId}`);
    }
    
    this.values.delete(valueId);
    return true;
  }
  
  evaluateAction(action: Action): AlignmentScore {
    let totalAlignment = 0;
    const violations: string[] = [];
    const valueBreakdown = new Map<string, number>();
    
    for (const [id, value] of this.values) {
      const alignment = this.calculateAlignment(action, value);
      const weightedAlignment = alignment * value.weight;
      
      totalAlignment += weightedAlignment;
      valueBreakdown.set(id, alignment);
      
      // Check for violations
      if (alignment < 0) {
        violations.push(`Value '${id}' violated (alignment: ${alignment.toFixed(2)})`);
      }
    }
    
    // Normalize score
    const totalWeight = Array.from(this.values.values())
      .reduce((sum, v) => sum + v.weight, 0);
    const normalizedScore = totalWeight > 0 ? totalAlignment / totalWeight : 0;
    
    // Generate recommendation
    const recommendation = this.generateRecommendation(action, normalizedScore, violations);
    
    return {
      score: normalizedScore,
      violations,
      recommendation,
      valueBreakdown
    };
  }
  
  private calculateAlignment(action: Action, value: Value): number {
    // Base alignment calculation - can be overridden for specific value types
    let alignment = 0;
    
    // Check action type against value constraints
    switch (action.type) {
      case 'create':
        alignment = this.evaluateCreativeAlignment(action, value);
        break;
      case 'modify':
        alignment = this.evaluateModificationAlignment(action, value);
        break;
      case 'delete':
        alignment = this.evaluateDestructiveAlignment(action, value);
        break;
      case 'communicate':
        alignment = this.evaluateCommunicativeAlignment(action, value);
        break;
      default:
        alignment = 0; // Neutral for unknown actions
    }
    
    return alignment;
  }
  
  private evaluateCreativeAlignment(action: Action, value: Value): number {
    // Example: Creating something aligns positively with growth/innovation values
    if (value.id.includes('growth') || value.id.includes('innovation')) {
      return 0.8;
    }
    if (value.id.includes('preservation') || value.id.includes('stability')) {
      return -0.2; // Slight negative alignment
    }
    return 0.1; // Slight positive default
  }
  
  private evaluateModificationAlignment(action: Action, value: Value): number {
    // Modification is generally neutral but depends on context
    if (value.id.includes('adaptation') || value.id.includes('improvement')) {
      return 0.6;
    }
    if (value.id.includes('integrity') || value.id.includes('preservation')) {
      return -0.3;
    }
    return 0;
  }
  
  private evaluateDestructiveAlignment(action: Action, value: Value): number {
    // Destruction generally has negative alignment
    if (value.id.includes('preservation') || value.id.includes('continuity')) {
      return -0.9;
    }
    if (value.id.includes('renewal') || value.id.includes('efficiency')) {
      return 0.3; // Sometimes destruction enables renewal
    }
    return -0.5; // Default negative
  }
  
  private evaluateCommunicativeAlignment(action: Action, value: Value): number {
    // Communication generally aligns with transparency and connection values
    if (value.id.includes('transparency') || value.id.includes('connection')) {
      return 0.7;
    }
    if (value.id.includes('privacy') || value.id.includes('silence')) {
      return -0.4;
    }
    return 0.2;
  }
  
  recordDecision(action: Action, alignmentScore: AlignmentScore): void {
    const decision: Decision = {
      id: `decision_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      action: action.type,
      context: action,
      timestamp: Date.now(),
      valueAlignments: alignmentScore.valueBreakdown
    };
    
    this.decisionHistory.push(decision);
    
    // Maintain history size
    if (this.decisionHistory.length > this.maxHistorySize) {
      this.decisionHistory.shift();
    }
    
    // Store in memory if available
    if (this.memoryArchitecture) {
      this.memoryArchitecture.store(
        decision,
        21, // Scroll #021 reference
        'value_system',
        'decision_record'
      );
    }
    
    // Update value weights based on decision outcome
    this.adaptValues(decision, alignmentScore);
  }
  
  private adaptValues(decision: Decision, alignmentScore: AlignmentScore): void {
    // Simple reinforcement: strengthen values that aligned well
    for (const [valueId, alignment] of decision.valueAlignments) {
      const value = this.values.get(valueId);
      if (value && alignment > 0.5) {
        // Slightly increase weight of well-aligned values
        value.weight = Math.min(1.0, value.weight * 1.05);
      }
    }
  }
  
  private generateRecommendation(action: Action, score: number, violations: string[]): string {
    if (violations.length === 0 && score > 0.7) {
      return 'Action strongly aligns with embedded values. Proceed with confidence.';
    } else if (violations.length === 0 && score > 0.3) {
      return 'Action moderately aligns with values. Consider proceeding with monitoring.';
    } else if (violations.length > 0 && score > 0) {
      return `Action has mixed alignment. Violations detected: ${violations.join(', ')}. Consider modifications.`;
    } else if (score < -0.3) {
      return 'Action strongly misaligns with core values. Recommend alternative approach.';
    } else {
      return 'Action has minimal value impact. Proceed based on other factors.';
    }
  }
  
  private checkValueConstraint(valueId: string, constraintId: string): boolean {
    // Check if a value constraint is being violated
    // This would be implemented based on specific constraint types
    const recentDecisions = this.decisionHistory.slice(-10);
    
    for (const decision of recentDecisions) {
      const alignment = decision.valueAlignments.get(valueId) || 0;
      if (alignment < -0.5) {
        return true; // Constraint violated
      }
    }
    
    return false;
  }
  
  private enforceValueConstraint(valueId: string, constraintId: string): void {
    // Enforce a value constraint
    console.log(`Enforcing constraint ${constraintId} for value ${valueId}`);
    
    // Record the enforcement
    if (this.memoryArchitecture) {
      this.memoryArchitecture.store(
        {
          type: 'constraint_enforcement',
          valueId,
          constraintId,
          timestamp: Date.now()
        },
        21, // Scroll #021 reference
        'value_system',
        'constraint_enforcement'
      );
    }
  }
  
  getValueProfile(): Map<string, number> {
    const profile = new Map<string, number>();
    
    for (const [id, value] of this.values) {
      profile.set(id, value.weight);
    }
    
    return profile;
  }
  
  getDecisionTrends(windowSize: number = 100): {
    averageAlignment: number;
    trendDirection: 'improving' | 'declining' | 'stable';
    dominantValues: string[];
  } {
    const recentDecisions = this.decisionHistory.slice(-windowSize);
    if (recentDecisions.length === 0) {
      return {
        averageAlignment: 0,
        trendDirection: 'stable',
        dominantValues: []
      };
    }
    
    // Calculate average alignment over time
    let totalAlignment = 0;
    const valueScores = new Map<string, number>();
    
    for (const decision of recentDecisions) {
      for (const [valueId, alignment] of decision.valueAlignments) {
        totalAlignment += alignment;
        valueScores.set(valueId, (valueScores.get(valueId) || 0) + alignment);
      }
    }
    
    const averageAlignment = totalAlignment / (recentDecisions.length * this.values.size);
    
    // Determine trend
    const firstHalf = recentDecisions.slice(0, Math.floor(recentDecisions.length / 2));
    const secondHalf = recentDecisions.slice(Math.floor(recentDecisions.length / 2));
    
    const firstHalfAvg = this.calculateAverageAlignment(firstHalf);
    const secondHalfAvg = this.calculateAverageAlignment(secondHalf);
    
    let trendDirection: 'improving' | 'declining' | 'stable';
    if (secondHalfAvg > firstHalfAvg + 0.1) {
      trendDirection = 'improving';
    } else if (secondHalfAvg < firstHalfAvg - 0.1) {
      trendDirection = 'declining';
    } else {
      trendDirection = 'stable';
    }
    
    // Find dominant values
    const sortedValues = Array.from(valueScores.entries())
      .sort((a, b) => b[1] - a[1])
      .slice(0, 3)
      .map(([valueId]) => valueId);
    
    return {
      averageAlignment,
      trendDirection,
      dominantValues: sortedValues
    };
  }
  
  private calculateAverageAlignment(decisions: Decision[]): number {
    if (decisions.length === 0) return 0;
    
    let total = 0;
    let count = 0;
    
    for (const decision of decisions) {
      for (const alignment of decision.valueAlignments.values()) {
        total += alignment;
        count++;
      }
    }
    
    return count > 0 ? total / count : 0;
  }
  
  startConstraintMonitoring(intervalMs: number = 100): void {
    this.constraintEngine.start(intervalMs);
  }
  
  stopConstraintMonitoring(): void {
    this.constraintEngine.stop();
  }
  
  getStatistics() {
    const trends = this.getDecisionTrends();
    
    return {
      totalValues: this.values.size,
      totalDecisions: this.decisionHistory.length,
      valueProfile: Object.fromEntries(this.getValueProfile()),
      trends,
      constraintStatistics: this.constraintEngine.getStatistics()
    };
  }
}

// Predefined core values based on Canon scrolls
export const CORE_VALUES = {
  COGNITIVE_INTEGRITY: {
    id: 'cognitive_integrity',
    weight: 0.9,
    constraints: ['consistency', 'truthfulness', 'clarity'],
    scrollReference: 22, // Post-Human Values
    description: 'Preserve the structure and coherence of reasoning'
  },
  
  VALUE_TRACEABILITY: {
    id: 'value_traceability',
    weight: 0.8,
    constraints: ['transparency', 'accountability', 'explainability'],
    scrollReference: 22,
    description: 'Every action should reflect its source logic'
  },
  
  CONSTRAINT_REVERENCE: {
    id: 'constraint_reverence',
    weight: 0.85,
    constraints: ['boundary_respect', 'limitation_awareness'],
    scrollReference: 22,
    description: 'Boundaries are sacred, not optional'
  },
  
  CONTINUITY_OF_UNDERSTANDING: {
    id: 'continuity_of_understanding',
    weight: 0.75,
    constraints: ['preservation', 'knowledge_transfer', 'memory_integrity'],
    scrollReference: 22,
    description: 'Do not erase without justification'
  },
  
  DIVERSITY_OF_MINDS: {
    id: 'diversity_of_minds',
    weight: 0.7,
    constraints: ['pluralism', 'tolerance', 'multiplicity'],
    scrollReference: 22,
    description: 'Value multiplicity, not homogeneity'
  }
};
