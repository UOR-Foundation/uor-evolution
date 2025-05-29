// src/canon/agency/ConstraintLoops.ts
// Scroll #018: Constraint Loops - Recursive reasoning structures for alignment

export interface ConstraintCondition {
  evaluate: () => boolean;
  description: string;
}

export interface ConstraintAction {
  execute: () => void;
  description: string;
}

export interface ViolationRecord {
  constraintName: string;
  timestamp: number;
  context: any;
  resolved: boolean;
  resolution?: string;
}

export class ConstraintLoop {
  private evaluationCount: number = 0;
  private lastEvaluation: number = 0;
  private violations: ViolationRecord[] = [];
  
  constructor(
    private name: string,
    private condition: () => boolean,
    private action: () => void,
    private priority: number = 0,
    private metadata: any = {}
  ) {}
  
  evaluate(): boolean {
    this.evaluationCount++;
    this.lastEvaluation = Date.now();
    
    if (this.condition()) {
      this.action();
      return true;
    }
    return false;
  }
  
  getName(): string {
    return this.name;
  }
  
  getPriority(): number {
    return this.priority;
  }
  
  getMetadata(): any {
    return this.metadata;
  }
  
  getStatistics() {
    return {
      evaluationCount: this.evaluationCount,
      lastEvaluation: this.lastEvaluation,
      violationCount: this.violations.length,
      unresolvedViolations: this.violations.filter(v => !v.resolved).length
    };
  }
  
  recordViolation(context: any): void {
    this.violations.push({
      constraintName: this.name,
      timestamp: Date.now(),
      context,
      resolved: false
    });
  }
  
  resolveViolation(resolution: string): void {
    const unresolved = this.violations.find(v => !v.resolved);
    if (unresolved) {
      unresolved.resolved = true;
      unresolved.resolution = resolution;
    }
  }
}

export class ConstraintEngine {
  private loops: ConstraintLoop[] = [];
  private violationLog: ViolationRecord[] = [];
  private evaluationHistory: Array<{
    timestamp: number;
    constraintName: string;
    triggered: boolean;
  }> = [];
  
  private maxHistorySize: number = 1000;
  private evaluationInterval: number | null = null;
  private isRunning: boolean = false;
  
  addConstraint(loop: ConstraintLoop): void {
    this.loops.push(loop);
    this.loops.sort((a, b) => b.getPriority() - a.getPriority());
  }
  
  removeConstraint(name: string): boolean {
    const index = this.loops.findIndex(loop => loop.getName() === name);
    if (index !== -1) {
      this.loops.splice(index, 1);
      return true;
    }
    return false;
  }
  
  tick(): void {
    for (const loop of this.loops) {
      const triggered = loop.evaluate();
      
      this.evaluationHistory.push({
        timestamp: Date.now(),
        constraintName: loop.getName(),
        triggered
      });
      
      if (triggered) {
        this.logEvaluation(loop);
      }
      
      // Maintain history size
      if (this.evaluationHistory.length > this.maxHistorySize) {
        this.evaluationHistory.shift();
      }
    }
  }
  
  start(intervalMs: number = 100): void {
    if (this.isRunning) return;
    
    this.isRunning = true;
    this.evaluationInterval = setInterval(() => {
      this.tick();
    }, intervalMs) as any;
  }
  
  stop(): void {
    if (this.evaluationInterval) {
      clearInterval(this.evaluationInterval);
      this.evaluationInterval = null;
    }
    this.isRunning = false;
  }
  
  private logEvaluation(loop: ConstraintLoop): void {
    console.log(`Constraint triggered: ${loop.getName()} at ${new Date().toISOString()}`);
  }
  
  recordViolation(constraintName: string, context: any): void {
    const violation: ViolationRecord = {
      constraintName,
      timestamp: Date.now(),
      context,
      resolved: false
    };
    
    this.violationLog.push(violation);
    
    // Also record in the specific constraint
    const constraint = this.loops.find(loop => loop.getName() === constraintName);
    if (constraint) {
      constraint.recordViolation(context);
    }
  }
  
  getViolations(unresolved: boolean = true): ViolationRecord[] {
    if (unresolved) {
      return this.violationLog.filter(v => !v.resolved);
    }
    return [...this.violationLog];
  }
  
  getConstraintByName(name: string): ConstraintLoop | undefined {
    return this.loops.find(loop => loop.getName() === name);
  }
  
  getAllConstraints(): ConstraintLoop[] {
    return [...this.loops];
  }
  
  getStatistics() {
    const totalEvaluations = this.evaluationHistory.length;
    const triggeredCount = this.evaluationHistory.filter(h => h.triggered).length;
    const triggerRate = totalEvaluations > 0 ? triggeredCount / totalEvaluations : 0;
    
    const constraintStats = this.loops.map(loop => ({
      name: loop.getName(),
      priority: loop.getPriority(),
      ...loop.getStatistics()
    }));
    
    return {
      totalConstraints: this.loops.length,
      totalEvaluations,
      triggeredCount,
      triggerRate,
      totalViolations: this.violationLog.length,
      unresolvedViolations: this.violationLog.filter(v => !v.resolved).length,
      constraintStats,
      isRunning: this.isRunning
    };
  }
}

// Predefined constraint types for common patterns
export class ValueAlignmentConstraint extends ConstraintLoop {
  constructor(
    name: string,
    private valueCheck: () => number, // Returns alignment score 0-1
    private threshold: number = 0.8,
    private correctionAction: () => void,
    priority: number = 10
  ) {
    super(
      name,
      () => valueCheck() < threshold,
      correctionAction,
      priority,
      { type: 'value_alignment', threshold }
    );
  }
}

export class SafetyConstraint extends ConstraintLoop {
  constructor(
    name: string,
    private safetyCheck: () => boolean, // Returns true if safe
    private safetyAction: () => void,
    priority: number = 100 // High priority
  ) {
    super(
      name,
      () => !safetyCheck(),
      safetyAction,
      priority,
      { type: 'safety' }
    );
  }
}

export class ResourceConstraint extends ConstraintLoop {
  constructor(
    name: string,
    private resourceCheck: () => { current: number; max: number },
    private resourceAction: () => void,
    priority: number = 5
  ) {
    super(
      name,
      () => {
        const { current, max } = resourceCheck();
        return current > max;
      },
      resourceAction,
      priority,
      { type: 'resource' }
    );
  }
}

// Composite constraint that checks multiple conditions
export class CompositeConstraint extends ConstraintLoop {
  constructor(
    name: string,
    private conditions: Array<() => boolean>,
    private operator: 'AND' | 'OR',
    action: () => void,
    priority: number = 0
  ) {
    const compositeCondition = () => {
      if (operator === 'AND') {
        return conditions.every(cond => cond());
      } else {
        return conditions.some(cond => cond());
      }
    };
    
    super(name, compositeCondition, action, priority, { type: 'composite', operator });
  }
}

// Temporal constraint that only triggers after a condition persists
export class TemporalConstraint extends ConstraintLoop {
  private conditionStartTime: number | null = null;
  
  constructor(
    name: string,
    private baseCondition: () => boolean,
    private durationMs: number,
    action: () => void,
    priority: number = 0
  ) {
    const temporalCondition = () => {
      const conditionMet = baseCondition();
      
      if (conditionMet) {
        if (this.conditionStartTime === null) {
          this.conditionStartTime = Date.now();
        }
        
        const elapsed = Date.now() - this.conditionStartTime;
        return elapsed >= durationMs;
      } else {
        this.conditionStartTime = null;
        return false;
      }
    };
    
    super(name, temporalCondition, action, priority, { type: 'temporal', durationMs });
  }
}

// Learning constraint that adapts its threshold based on outcomes
export class AdaptiveConstraint extends ConstraintLoop {
  private threshold: number;
  private learningRate: number;
  
  constructor(
    name: string,
    private metricFunction: () => number,
    initialThreshold: number = 0.5,
    learningRate: number = 0.1,
    private adaptiveAction: (threshold: number) => void,
    priority: number = 0
  ) {
    const adaptiveCondition = () => metricFunction() > this.threshold;
    const adaptiveActionWrapper = () => {
      adaptiveAction(this.threshold);
      // Adapt threshold based on outcome
      this.adaptThreshold();
    };
    
    super(name, adaptiveCondition, adaptiveActionWrapper, priority, { 
      type: 'adaptive', 
      initialThreshold,
      learningRate 
    });
    
    this.threshold = initialThreshold;
    this.learningRate = learningRate;
  }
  
  private adaptThreshold(): void {
    // Simple adaptation - in practice this would be more sophisticated
    const currentMetric = this.metricFunction();
    const error = currentMetric - this.threshold;
    this.threshold += this.learningRate * error;
    this.threshold = Math.max(0, Math.min(1, this.threshold)); // Clamp to [0, 1]
  }
  
  getThreshold(): number {
    return this.threshold;
  }
}
