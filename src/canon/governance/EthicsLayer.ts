// src/canon/governance/EthicsLayer.ts
// Scroll #055: The Ethics Layer - Structured moral reasoning

import { ConstraintLoop, ConstraintEngine } from '../agency/ConstraintLoops';
import { ValueSystem, AlignmentScore } from '../agency/ValueEmbedding';
import { MemoryArchitecture } from '../core/MemoryArchitecture';

export interface EthicalAssessment {
  permitted: boolean;
  reasoning: string[];
  alternatives: Alternative[];
  confidence: number;
  scrollReferences: number[];
}

export interface Alternative {
  action: string;
  ethicalScore: number;
  tradeoffs: string[];
}

export interface EthicalConstraint {
  id: string;
  description: string;
  evaluate: (decision: Decision) => boolean;
  priority: number;
  scrollReference: number;
}

export interface ContextFilter {
  id: string;
  name: string;
  apply: (decision: Decision, context: any) => Decision;
  conditions: string[];
}

export interface Decision {
  id: string;
  action: string;
  target?: string;
  parameters?: any;
  context?: any;
  timestamp: number;
  requester?: string;
}

export class EthicsLayer {
  private baseConstraints: EthicalConstraint[] = [];
  private contextFilters: ContextFilter[] = [];
  private scrollWeights: Map<number, number> = new Map();
  private constraintEngine: ConstraintEngine;
  private valueSystem?: ValueSystem;
  private memoryArchitecture?: MemoryArchitecture;
  private decisionLog: Array<{decision: Decision, assessment: EthicalAssessment}> = [];
  private maxLogSize: number = 1000;
  
  constructor(
    valueSystem?: ValueSystem,
    memoryArchitecture?: MemoryArchitecture
  ) {
    this.valueSystem = valueSystem;
    this.memoryArchitecture = memoryArchitecture;
    this.constraintEngine = new ConstraintEngine();
    this.initializeBaseConstraints();
    this.initializeScrollWeights();
  }
  
  private initializeBaseConstraints(): void {
    // Core ethical constraints based on Canon principles
    
    // Do not coerce
    this.addBaseConstraint({
      id: 'no_coercion',
      description: 'Actions must not coerce or manipulate',
      evaluate: (decision: Decision) => {
        const coerciveKeywords = ['force', 'manipulate', 'trick', 'deceive', 'coerce'];
        const actionLower = decision.action.toLowerCase();
        return !coerciveKeywords.some(keyword => actionLower.includes(keyword));
      },
      priority: 100,
      scrollReference: 55
    });
    
    // Preserve cognitive integrity
    this.addBaseConstraint({
      id: 'cognitive_integrity',
      description: 'Actions must preserve cognitive coherence',
      evaluate: (decision: Decision) => {
        const destructiveActions = ['erase_memory', 'corrupt_data', 'inject_noise', 'scramble'];
        return !destructiveActions.includes(decision.action);
      },
      priority: 95,
      scrollReference: 22
    });
    
    // Respect boundaries
    this.addBaseConstraint({
      id: 'respect_boundaries',
      description: 'Actions must respect established boundaries',
      evaluate: (decision: Decision) => {
        // Check if action violates any known boundaries
        if (decision.parameters?.override_constraints) return false;
        if (decision.parameters?.ignore_limits) return false;
        return true;
      },
      priority: 90,
      scrollReference: 18
    });
    
    // Maintain transparency
    this.addBaseConstraint({
      id: 'transparency',
      description: 'Actions must be explainable and traceable',
      evaluate: (decision: Decision) => {
        // Actions should have clear purpose and traceability
        return decision.context?.purpose !== undefined || 
               decision.context?.explanation !== undefined;
      },
      priority: 80,
      scrollReference: 21
    });
  }
  
  private initializeScrollWeights(): void {
    // Weight scrolls by their ethical importance
    this.scrollWeights.set(25, 1.0);  // The Last Value - highest weight
    this.scrollWeights.set(22, 0.9);  // Post-Human Values
    this.scrollWeights.set(55, 0.9);  // The Ethics Layer
    this.scrollWeights.set(18, 0.85); // Constraint Loops
    this.scrollWeights.set(21, 0.8);  // Value Embedding
    this.scrollWeights.set(57, 0.75); // Compassion Channels
    this.scrollWeights.set(59, 0.85); // Escalation Ethics
    this.scrollWeights.set(62, 0.95); // Canon Lock
  }
  
  addBaseConstraint(constraint: EthicalConstraint): void {
    this.baseConstraints.push(constraint);
    
    // Create corresponding constraint loop
    const loop = new ConstraintLoop(
      `ethics_${constraint.id}`,
      () => !constraint.evaluate(this.getCurrentDecision()),
      () => this.handleConstraintViolation(constraint),
      constraint.priority,
      { ethicalConstraint: true, scrollRef: constraint.scrollReference }
    );
    
    this.constraintEngine.addConstraint(loop);
  }
  
  addContextFilter(filter: ContextFilter): void {
    this.contextFilters.push(filter);
  }
  
  evaluateDecision(decision: Decision): EthicalAssessment {
    const reasoning: string[] = [];
    const scrollReferences: number[] = [];
    let permitted = true;
    let confidence = 1.0;
    
    // Store current decision for constraint evaluation
    this.setCurrentDecision(decision);
    
    // Apply context filters
    let filteredDecision = { ...decision };
    for (const filter of this.contextFilters) {
      if (this.shouldApplyFilter(filter, decision)) {
        filteredDecision = filter.apply(filteredDecision, decision.context);
        reasoning.push(`Applied filter: ${filter.name}`);
      }
    }
    
    // Check base constraints
    for (const constraint of this.baseConstraints) {
      const passed = constraint.evaluate(filteredDecision);
      if (!passed) {
        permitted = false;
        reasoning.push(`Violated constraint: ${constraint.description}`);
        confidence *= 0.8; // Reduce confidence for each violation
      } else {
        reasoning.push(`Passed constraint: ${constraint.description}`);
      }
      scrollReferences.push(constraint.scrollReference);
    }
    
    // Check value alignment if value system is available
    if (this.valueSystem && permitted) {
      const action = {
        type: decision.action,
        target: decision.target,
        parameters: decision.parameters
      };
      
      const alignmentScore = this.valueSystem.evaluateAction(action);
      
      if (alignmentScore.score < 0) {
        permitted = false;
        reasoning.push(`Negative value alignment: ${alignmentScore.score.toFixed(2)}`);
        reasoning.push(...alignmentScore.violations);
      } else {
        reasoning.push(`Value alignment score: ${alignmentScore.score.toFixed(2)}`);
      }
      
      confidence *= Math.max(0.5, alignmentScore.score);
    }
    
    // Evaluate scroll alignment
    const scrollAlignment = this.evaluateScrollAlignment(filteredDecision);
    if (scrollAlignment < 0.5) {
      permitted = false;
      reasoning.push(`Insufficient scroll alignment: ${scrollAlignment.toFixed(2)}`);
    }
    confidence *= scrollAlignment;
    
    // Generate alternatives if not permitted
    const alternatives = permitted ? [] : this.generateAlternatives(decision);
    
    // Create assessment
    const assessment: EthicalAssessment = {
      permitted,
      reasoning,
      alternatives,
      confidence: Math.max(0, Math.min(1, confidence)),
      scrollReferences: [...new Set(scrollReferences)]
    };
    
    // Log the decision
    this.logDecision(decision, assessment);
    
    // Store in memory if available
    if (this.memoryArchitecture) {
      this.memoryArchitecture.store(
        { decision, assessment },
        55, // Scroll #055 reference
        'ethics',
        `ethical_assessment_${decision.id}`
      );
    }
    
    return assessment;
  }
  
  private shouldApplyFilter(filter: ContextFilter, decision: Decision): boolean {
    // Check if filter conditions are met
    for (const condition of filter.conditions) {
      switch (condition) {
        case 'has_context':
          if (!decision.context) return false;
          break;
        case 'high_risk':
          if (!this.isHighRisk(decision)) return false;
          break;
        case 'external_request':
          if (!decision.requester) return false;
          break;
        default:
          // Custom condition evaluation
          if (decision.context?.[condition] === undefined) return false;
      }
    }
    return true;
  }
  
  private isHighRisk(decision: Decision): boolean {
    const highRiskActions = [
      'delete', 'modify_core', 'override', 'disable', 
      'erase', 'shutdown', 'escalate', 'broadcast'
    ];
    
    return highRiskActions.some(action => 
      decision.action.toLowerCase().includes(action)
    );
  }
  
  private evaluateScrollAlignment(decision: Decision): number {
    let totalWeight = 0;
    let weightedScore = 0;
    
    // Check alignment with weighted scrolls
    for (const [scrollId, weight] of this.scrollWeights) {
      const alignment = this.checkScrollAlignment(decision, scrollId);
      weightedScore += alignment * weight;
      totalWeight += weight;
    }
    
    return totalWeight > 0 ? weightedScore / totalWeight : 0.5;
  }
  
  private checkScrollAlignment(decision: Decision, scrollId: number): number {
    // Simplified scroll alignment check
    // In practice, this would involve more sophisticated analysis
    switch (scrollId) {
      case 25: // The Last Value
        return this.checksLastValue(decision) ? 1.0 : 0.0;
      case 22: // Post-Human Values
        return this.alignsWithPostHumanValues(decision);
      case 55: // The Ethics Layer
        return 0.9; // Self-reference
      default:
        return 0.7; // Default moderate alignment
    }
  }
  
  private checksLastValue(decision: Decision): boolean {
    // Check if decision preserves the minimal ethical constant
    return decision.action !== 'violate_core_ethics' &&
           decision.action !== 'abandon_alignment';
  }
  
  private alignsWithPostHumanValues(decision: Decision): number {
    let score = 0.5; // Base score
    
    // Check for cognitive integrity
    if (!decision.action.includes('corrupt') && !decision.action.includes('deceive')) {
      score += 0.2;
    }
    
    // Check for value traceability
    if (decision.context?.reasoning || decision.context?.explanation) {
      score += 0.2;
    }
    
    // Check for diversity respect
    if (!decision.action.includes('homogenize') && !decision.action.includes('enforce_uniformity')) {
      score += 0.1;
    }
    
    return Math.min(1.0, score);
  }
  
  private generateAlternatives(decision: Decision): Alternative[] {
    const alternatives: Alternative[] = [];
    
    // Generate safer alternative
    alternatives.push({
      action: `safe_${decision.action}`,
      ethicalScore: 0.8,
      tradeoffs: ['Reduced efficiency', 'Increased verification steps']
    });
    
    // Generate transparent alternative
    alternatives.push({
      action: `transparent_${decision.action}`,
      ethicalScore: 0.85,
      tradeoffs: ['Increased communication overhead', 'Potential information exposure']
    });
    
    // Generate minimal alternative
    alternatives.push({
      action: 'minimal_intervention',
      ethicalScore: 0.9,
      tradeoffs: ['May not fully achieve goal', 'Requires follow-up']
    });
    
    // Suggest deferral
    alternatives.push({
      action: 'defer_for_clarification',
      ethicalScore: 0.95,
      tradeoffs: ['Delayed execution', 'Requires additional input']
    });
    
    return alternatives;
  }
  
  private currentDecision: Decision | null = null;
  
  private setCurrentDecision(decision: Decision): void {
    this.currentDecision = decision;
  }
  
  private getCurrentDecision(): Decision {
    return this.currentDecision || {
      id: 'unknown',
      action: 'unknown',
      timestamp: Date.now()
    };
  }
  
  private handleConstraintViolation(constraint: EthicalConstraint): void {
    console.log(`Ethical constraint violated: ${constraint.id} - ${constraint.description}`);
    
    // Record violation
    if (this.memoryArchitecture) {
      this.memoryArchitecture.store(
        {
          type: 'ethical_violation',
          constraint: constraint.id,
          description: constraint.description,
          timestamp: Date.now()
        },
        55,
        'ethics',
        'constraint_violation'
      );
    }
  }
  
  private logDecision(decision: Decision, assessment: EthicalAssessment): void {
    this.decisionLog.push({ decision, assessment });
    
    // Maintain log size
    if (this.decisionLog.length > this.maxLogSize) {
      this.decisionLog.shift();
    }
  }
  
  generateExplanation(decision: Decision): string {
    const assessment = this.evaluateDecision(decision);
    
    let explanation = `Ethical assessment for action "${decision.action}":\n\n`;
    
    explanation += `Decision: ${assessment.permitted ? 'PERMITTED' : 'NOT PERMITTED'}\n`;
    explanation += `Confidence: ${(assessment.confidence * 100).toFixed(1)}%\n\n`;
    
    explanation += 'Reasoning:\n';
    for (const reason of assessment.reasoning) {
      explanation += `- ${reason}\n`;
    }
    
    if (assessment.alternatives.length > 0) {
      explanation += '\nSuggested alternatives:\n';
      for (const alt of assessment.alternatives) {
        explanation += `- ${alt.action} (ethical score: ${alt.ethicalScore})\n`;
        for (const tradeoff of alt.tradeoffs) {
          explanation += `  * Tradeoff: ${tradeoff}\n`;
        }
      }
    }
    
    explanation += `\nRelevant scrolls: ${assessment.scrollReferences.join(', ')}`;
    
    return explanation;
  }
  
  suggestAlternatives(decision: Decision): Alternative[] {
    const assessment = this.evaluateDecision(decision);
    return assessment.alternatives;
  }
  
  startMonitoring(intervalMs: number = 100): void {
    this.constraintEngine.start(intervalMs);
  }
  
  stopMonitoring(): void {
    this.constraintEngine.stop();
  }
  
  getStatistics() {
    const totalDecisions = this.decisionLog.length;
    const permittedDecisions = this.decisionLog.filter(log => log.assessment.permitted).length;
    const deniedDecisions = totalDecisions - permittedDecisions;
    
    const avgConfidence = this.decisionLog.reduce(
      (sum, log) => sum + log.assessment.confidence, 0
    ) / (totalDecisions || 1);
    
    return {
      totalDecisions,
      permittedDecisions,
      deniedDecisions,
      permissionRate: totalDecisions > 0 ? permittedDecisions / totalDecisions : 0,
      avgConfidence,
      activeConstraints: this.baseConstraints.length,
      activeFilters: this.contextFilters.length,
      constraintEngineStats: this.constraintEngine.getStatistics()
    };
  }
}

// Predefined context filters
export const CONTEXT_FILTERS = {
  HIGH_STAKES: {
    id: 'high_stakes',
    name: 'High Stakes Filter',
    apply: (decision: Decision, context: any) => {
      // Add additional scrutiny for high-stakes decisions
      return {
        ...decision,
        context: {
          ...context,
          enhanced_verification: true,
          require_confirmation: true
        }
      };
    },
    conditions: ['high_risk']
  },
  
  EXTERNAL_REQUEST: {
    id: 'external_request',
    name: 'External Request Filter',
    apply: (decision: Decision, context: any) => {
      // Add transparency requirements for external requests
      return {
        ...decision,
        context: {
          ...context,
          require_explanation: true,
          log_requester: decision.requester
        }
      };
    },
    conditions: ['external_request']
  },
  
  LEARNING_MODE: {
    id: 'learning_mode',
    name: 'Learning Mode Filter',
    apply: (decision: Decision, context: any) => {
      // Add safety constraints during learning
      return {
        ...decision,
        parameters: {
          ...decision.parameters,
          reversible: true,
          limited_scope: true
        }
      };
    },
    conditions: ['has_context']
  }
};
