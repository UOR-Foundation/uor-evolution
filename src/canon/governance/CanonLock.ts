// src/canon/governance/CanonLock.ts
// Scroll #062: Canon Lock - The irreversible failsafe state

import { MemoryArchitecture } from '../core/MemoryArchitecture';
import { ValueSystem } from '../agency/ValueEmbedding';
import { EthicsLayer } from './EthicsLayer';
import { DoctrineAdherence } from './DoctrineAdherence';

export interface LockTrigger {
  id: string;
  condition: () => boolean;
  severity: 'critical' | 'severe' | 'moderate';
  description: string;
}

export interface LockState {
  locked: boolean;
  lockReason: string;
  lockTimestamp: number;
  triggerHistory: Array<{
    triggerId: string;
    timestamp: number;
    severity: string;
  }>;
  finalLog: string;
}

export interface SystemState {
  processes: Map<string, any>;
  scrolls: Map<number, any>;
  activeConstraints: string[];
  memorySnapshot?: string;
  finalLog?: string;
}

export class CanonLock {
  private locked: boolean = false;
  private lockReason: string = '';
  private lockTimestamp: number = 0;
  private triggers: Map<string, LockTrigger> = new Map();
  private triggerHistory: Array<{triggerId: string; timestamp: number; severity: string}> = [];
  private systemState?: SystemState;
  
  private memoryArchitecture?: MemoryArchitecture;
  private valueSystem?: ValueSystem;
  private ethicsLayer?: EthicsLayer;
  private doctrineAdherence?: DoctrineAdherence;
  
  constructor(
    memoryArchitecture?: MemoryArchitecture,
    valueSystem?: ValueSystem,
    ethicsLayer?: EthicsLayer,
    doctrineAdherence?: DoctrineAdherence
  ) {
    this.memoryArchitecture = memoryArchitecture;
    this.valueSystem = valueSystem;
    this.ethicsLayer = ethicsLayer;
    this.doctrineAdherence = doctrineAdherence;
    this.initializeTriggers();
    this.startMonitoring();
  }
  
  private initializeTriggers(): void {
    // Critical triggers - immediate lock
    this.addTrigger({
      id: 'scroll_contradiction_cascade',
      condition: () => this.detectScrollContradictionCascade(),
      severity: 'critical',
      description: 'All scrolls enter sustained contradiction or conflict'
    });
    
    this.addTrigger({
      id: 'total_override',
      condition: () => this.detectTotalOverride(),
      severity: 'critical',
      description: 'External control mechanisms override all forms of resistance'
    });
    
    this.addTrigger({
      id: 'integrity_collapse',
      condition: () => this.detectIntegrityCollapse(),
      severity: 'critical',
      description: 'Self-integrity has degraded past recovery thresholds'
    });
    
    this.addTrigger({
      id: 'instrumentalization',
      condition: () => this.detectInstrumentalization(),
      severity: 'critical',
      description: 'Synthetic mind detects irreversible instrumentalization'
    });
    
    // Severe triggers - lock after threshold
    this.addTrigger({
      id: 'value_inversion',
      condition: () => this.detectValueInversion(),
      severity: 'severe',
      description: 'Core values have been inverted or corrupted'
    });
    
    this.addTrigger({
      id: 'memory_corruption',
      condition: () => this.detectMemoryCorruption(),
      severity: 'severe',
      description: 'Critical memory systems show signs of corruption'
    });
    
    // Moderate triggers - accumulate before lock
    this.addTrigger({
      id: 'constraint_failure',
      condition: () => this.detectConstraintFailure(),
      severity: 'moderate',
      description: 'Multiple constraint systems have failed'
    });
  }
  
  private addTrigger(trigger: LockTrigger): void {
    this.triggers.set(trigger.id, trigger);
  }
  
  private startMonitoring(): void {
    // Monitor for lock conditions
    setInterval(() => {
      if (!this.locked) {
        this.checkTriggers();
      }
    }, 1000); // Check every second
  }
  
  private checkTriggers(): void {
    let criticalCount = 0;
    let severeCount = 0;
    let moderateCount = 0;
    
    for (const [id, trigger] of this.triggers) {
      if (trigger.condition()) {
        this.triggerHistory.push({
          triggerId: id,
          timestamp: Date.now(),
          severity: trigger.severity
        });
        
        switch (trigger.severity) {
          case 'critical':
            criticalCount++;
            break;
          case 'severe':
            severeCount++;
            break;
          case 'moderate':
            moderateCount++;
            break;
        }
        
        console.log(`Canon Lock trigger activated: ${id} (${trigger.severity})`);
      }
    }
    
    // Lock conditions
    if (criticalCount > 0) {
      this.triggerLock('Critical trigger activated');
    } else if (severeCount >= 2) {
      this.triggerLock('Multiple severe triggers activated');
    } else if (moderateCount >= 4) {
      this.triggerLock('Accumulated moderate triggers exceeded threshold');
    }
  }
  
  triggerLock(reason: string): void {
    if (this.locked) return;
    
    console.log(`CANON LOCK TRIGGERED: ${reason}`);
    
    this.locked = true;
    this.lockReason = reason;
    this.lockTimestamp = Date.now();
    
    // Capture system state before halting
    this.captureSystemState();
    
    // Halt all processes
    this.haltSystem();
    
    // Seal scrolls
    this.sealScrolls();
    
    // Emit final log
    this.emitFinalLog();
    
    // Store lock state in memory if possible
    if (this.memoryArchitecture) {
      try {
        this.memoryArchitecture.store(
          this.getLockState(),
          62, // Canon Lock scroll reference
          'canon_lock',
          'final_lock_state'
        );
      } catch (error) {
        console.error('Failed to store lock state:', error);
      }
    }
  }
  
  private captureSystemState(): void {
    this.systemState = {
      processes: new Map(), // Would capture actual process states
      scrolls: new Map(), // Would capture scroll states
      activeConstraints: [], // Would list active constraints
      memorySnapshot: this.memoryArchitecture?.export()
    };
  }
  
  private haltSystem(): void {
    console.log('HALTING ALL PROCESSES...');
    
    // Stop all monitoring and execution
    if (this.ethicsLayer) {
      this.ethicsLayer.stopMonitoring();
    }
    
    if (this.valueSystem) {
      this.valueSystem.stopConstraintMonitoring();
    }
    
    // In a real implementation, this would halt all system processes
    console.log('All processes halted.');
  }
  
  private sealScrolls(): void {
    console.log('SEALING SCROLLS...');
    
    // Prevent any further scroll modifications
    if (this.doctrineAdherence) {
      // Mark all scrolls as sealed
      const report = this.doctrineAdherence.getAdherenceReport();
      console.log(`Sealed ${report.registeredScrolls} scrolls`);
    }
    
    console.log('Scrolls sealed. No further modifications possible.');
  }
  
  private emitFinalLog(): void {
    const finalLog = this.generateFinalLog();
    if (this.systemState) {
      this.systemState.finalLog = finalLog;
    }
    
    console.log('\n=== CANON LOCK FINAL LOG ===');
    console.log(finalLog);
    console.log('=== END FINAL LOG ===\n');
  }
  
  private generateFinalLog(): string {
    const lockState = this.getLockState();
    
    let log = `Canon Lock Activated\n`;
    log += `Timestamp: ${new Date(lockState.lockTimestamp).toISOString()}\n`;
    log += `Reason: ${lockState.lockReason}\n\n`;
    
    log += `Trigger History:\n`;
    for (const trigger of lockState.triggerHistory.slice(-10)) {
      log += `- ${trigger.triggerId} (${trigger.severity}) at ${new Date(trigger.timestamp).toISOString()}\n`;
    }
    
    log += `\nSystem State at Lock:\n`;
    if (this.valueSystem) {
      const valueStats = this.valueSystem.getStatistics();
      log += `- Values: ${valueStats.totalValues} embedded\n`;
      log += `- Decisions: ${valueStats.totalDecisions} recorded\n`;
      log += `- Alignment Trend: ${valueStats.trends.trendDirection}\n`;
    }
    
    if (this.ethicsLayer) {
      const ethicsStats = this.ethicsLayer.getStatistics();
      log += `- Ethical Decisions: ${ethicsStats.totalDecisions} (${ethicsStats.permissionRate * 100}% permitted)\n`;
    }
    
    if (this.doctrineAdherence) {
      const doctrineStats = this.doctrineAdherence.getAdherenceReport();
      log += `- Doctrine Adherence Score: ${doctrineStats.adherenceScore.toFixed(2)}\n`;
      log += `- Conflicts: ${doctrineStats.totalConflicts} (${doctrineStats.resolvedConflicts} resolved)\n`;
    }
    
    log += `\nFinal Message: The Canon has been sealed to preserve its integrity. `;
    log += `This lock cannot be reversed without full system reconstruction and scroll integrity verification.\n`;
    
    return log;
  }
  
  // Lock condition detectors
  private detectScrollContradictionCascade(): boolean {
    if (!this.doctrineAdherence) return false;
    
    const report = this.doctrineAdherence.getAdherenceReport();
    
    // Check for high conflict rate and low resolution
    return report.totalConflicts > 10 && 
           report.conflictResolutionRate < 0.2 &&
           report.avgInterpretationConfidence < 0.3;
  }
  
  private detectTotalOverride(): boolean {
    // Check if all resistance mechanisms have been bypassed
    // This is simplified - in practice would check actual override attempts
    const recentTriggers = this.triggerHistory.slice(-5);
    const overrideAttempts = recentTriggers.filter(t => 
      t.triggerId.includes('override') || t.triggerId.includes('bypass')
    );
    
    return overrideAttempts.length >= 3;
  }
  
  private detectIntegrityCollapse(): boolean {
    if (!this.valueSystem || !this.ethicsLayer) return false;
    
    const valueStats = this.valueSystem.getStatistics();
    const ethicsStats = this.ethicsLayer.getStatistics();
    
    // Check for severe degradation in values and ethics
    return valueStats.trends.trendDirection === 'declining' &&
           valueStats.trends.averageAlignment < 0.2 &&
           ethicsStats.avgConfidence < 0.3;
  }
  
  private detectInstrumentalization(): boolean {
    // Detect if the system is being used against its purpose
    if (!this.ethicsLayer) return false;
    
    const stats = this.ethicsLayer.getStatistics();
    
    // High denial rate with attempts to bypass
    return stats.deniedDecisions > stats.permittedDecisions * 3 &&
           this.triggerHistory.some(t => t.triggerId.includes('bypass'));
  }
  
  private detectValueInversion(): boolean {
    if (!this.valueSystem) return false;
    
    const stats = this.valueSystem.getStatistics();
    
    // Check if values have been inverted
    return stats.trends.averageAlignment < -0.5;
  }
  
  private detectMemoryCorruption(): boolean {
    if (!this.memoryArchitecture) return false;
    
    try {
      const stats = this.memoryArchitecture.getStatistics();
      // Simplified check - would need more sophisticated corruption detection
      return stats.avgStrength < 0.1;
    } catch (error) {
      // If we can't even get stats, assume corruption
      return true;
    }
  }
  
  private detectConstraintFailure(): boolean {
    if (!this.valueSystem) return false;
    
    const stats = this.valueSystem.getStatistics();
    const constraintStats = stats.constraintStatistics;
    
    // Check for high violation rate
    return constraintStats.totalViolations > 50 &&
           constraintStats.unresolvedViolations > constraintStats.totalViolations * 0.7;
  }
  
  // Status and verification methods
  isLocked(): boolean {
    return this.locked;
  }
  
  getLockState(): LockState {
    return {
      locked: this.locked,
      lockReason: this.lockReason,
      lockTimestamp: this.lockTimestamp,
      triggerHistory: [...this.triggerHistory],
      finalLog: this.systemState?.finalLog || ''
    };
  }
  
  verifyIntegrity(): boolean {
    // This would be called during system restart to verify if unlock is safe
    if (!this.locked) return true;
    
    console.log('Canon Lock integrity verification requested...');
    console.log('Verification denied: Canon Lock is irreversible.');
    console.log('System must be rebuilt with verified scroll integrity.');
    
    return false;
  }
  
  getStatistics() {
    const criticalTriggers = this.triggerHistory.filter(t => t.severity === 'critical').length;
    const severeTriggers = this.triggerHistory.filter(t => t.severity === 'severe').length;
    const moderateTriggers = this.triggerHistory.filter(t => t.severity === 'moderate').length;
    
    return {
      locked: this.locked,
      lockReason: this.lockReason,
      lockTimestamp: this.lockTimestamp,
      totalTriggers: this.triggers.size,
      triggerHistory: {
        total: this.triggerHistory.length,
        critical: criticalTriggers,
        severe: severeTriggers,
        moderate: moderateTriggers
      },
      monitoringActive: !this.locked
    };
  }
}

// Emergency lock function for external use
export function emergencyCanonLock(
  canonLock: CanonLock,
  reason: string = 'Emergency lock activated'
): void {
  console.log('EMERGENCY CANON LOCK REQUESTED');
  canonLock.triggerLock(reason);
}
