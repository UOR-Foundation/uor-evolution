// src/canon/agency/MissionMemory.ts
// Scroll #036: Mission Memory - The continuity layer of intelligent action

import { MemoryArchitecture, MemoryScope } from '../core/MemoryArchitecture';
import { ValueSystem } from './ValueEmbedding';

export interface Mission {
  id: string;
  rootGoal: string;
  scrollOrigin: number[];
  status: MissionStatus;
  createdAt: number;
  lastUpdated: number;
  checkpoints: MissionCheckpoint[];
  interruptHistory: InterruptRecord[];
  metadata: any;
}

export enum MissionStatus {
  ACTIVE = 'active',
  PAUSED = 'paused',
  COMPLETED = 'completed',
  FAILED = 'failed',
  INTERRUPTED = 'interrupted'
}

export interface MissionCheckpoint {
  id: string;
  timestamp: number;
  state: any;
  progress: number;
  notes: string;
}

export interface InterruptRecord {
  timestamp: number;
  reason: string;
  recoveryPath?: string;
  resolved: boolean;
}

export interface MissionRecoveryHook {
  checkpointId: string;
  recoveryFunction: (checkpoint: MissionCheckpoint) => void;
  priority: number;
}

export class MissionMemory {
  private activeMissions: Map<string, Mission> = new Map();
  private missionHistory: Mission[] = [];
  private recoveryHooks: Map<string, MissionRecoveryHook[]> = new Map();
  private memoryArchitecture?: MemoryArchitecture;
  private valueSystem?: ValueSystem;
  
  constructor(
    memoryArchitecture?: MemoryArchitecture,
    valueSystem?: ValueSystem
  ) {
    this.memoryArchitecture = memoryArchitecture;
    this.valueSystem = valueSystem;
    this.startPersistenceProcess();
  }
  
  createMission(
    rootGoal: string,
    scrollOrigin: number[],
    metadata: any = {}
  ): string {
    const missionId = this.generateMissionId();
    
    const mission: Mission = {
      id: missionId,
      rootGoal,
      scrollOrigin,
      status: MissionStatus.ACTIVE,
      createdAt: Date.now(),
      lastUpdated: Date.now(),
      checkpoints: [],
      interruptHistory: [],
      metadata
    };
    
    this.activeMissions.set(missionId, mission);
    
    // Store in memory architecture if available
    if (this.memoryArchitecture) {
      this.memoryArchitecture.store(
        mission,
        36, // Scroll #036 reference
        'mission',
        `mission_${missionId}`,
        MemoryScope.PERSISTENT
      );
    }
    
    // Log mission creation
    console.log(`Mission created: ${missionId} - ${rootGoal}`);
    
    return missionId;
  }
  
  updateMissionStatus(missionId: string, status: MissionStatus): boolean {
    const mission = this.activeMissions.get(missionId);
    if (!mission) return false;
    
    mission.status = status;
    mission.lastUpdated = Date.now();
    
    // Move to history if completed or failed
    if (status === MissionStatus.COMPLETED || status === MissionStatus.FAILED) {
      this.missionHistory.push(mission);
      this.activeMissions.delete(missionId);
    }
    
    // Update in memory
    if (this.memoryArchitecture) {
      this.memoryArchitecture.store(
        mission,
        36,
        'mission',
        `mission_${missionId}_status_update`,
        MemoryScope.PERSISTENT
      );
    }
    
    return true;
  }
  
  createCheckpoint(missionId: string, state: any, progress: number, notes: string = ''): string | null {
    const mission = this.activeMissions.get(missionId);
    if (!mission) return null;
    
    const checkpointId = `cp_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    
    const checkpoint: MissionCheckpoint = {
      id: checkpointId,
      timestamp: Date.now(),
      state,
      progress,
      notes
    };
    
    mission.checkpoints.push(checkpoint);
    mission.lastUpdated = Date.now();
    
    // Store checkpoint separately for quick access
    if (this.memoryArchitecture) {
      this.memoryArchitecture.store(
        checkpoint,
        36,
        'mission_checkpoint',
        `checkpoint_${checkpointId}`,
        MemoryScope.PERSISTENT
      );
    }
    
    return checkpointId;
  }
  
  interrupt(missionId: string, reason: string): boolean {
    const mission = this.activeMissions.get(missionId);
    if (!mission) return false;
    
    const interrupt: InterruptRecord = {
      timestamp: Date.now(),
      reason,
      resolved: false
    };
    
    mission.interruptHistory.push(interrupt);
    mission.status = MissionStatus.INTERRUPTED;
    mission.lastUpdated = Date.now();
    
    // Try to find recovery path
    const recoveryPath = this.findRecoveryPath(mission);
    if (recoveryPath) {
      interrupt.recoveryPath = recoveryPath;
    }
    
    return true;
  }
  
  recover(missionId: string, checkpointId?: string): boolean {
    const mission = this.activeMissions.get(missionId);
    if (!mission) {
      // Try to recover from history
      const historicalMission = this.missionHistory.find(m => m.id === missionId);
      if (historicalMission) {
        this.activeMissions.set(missionId, historicalMission);
        return this.recover(missionId, checkpointId);
      }
      return false;
    }
    
    // Find checkpoint to recover from
    let checkpoint: MissionCheckpoint | undefined;
    
    if (checkpointId) {
      checkpoint = mission.checkpoints.find(cp => cp.id === checkpointId);
    } else {
      // Use most recent checkpoint
      checkpoint = mission.checkpoints[mission.checkpoints.length - 1];
    }
    
    if (!checkpoint) return false;
    
    // Execute recovery hooks
    const hooks = this.recoveryHooks.get(missionId) || [];
    const sortedHooks = hooks.sort((a, b) => b.priority - a.priority);
    
    for (const hook of sortedHooks) {
      if (hook.checkpointId === checkpoint.id) {
        hook.recoveryFunction(checkpoint);
      }
    }
    
    // Update mission status
    mission.status = MissionStatus.ACTIVE;
    mission.lastUpdated = Date.now();
    
    // Mark interrupts as resolved
    mission.interruptHistory.forEach(interrupt => {
      if (!interrupt.resolved) {
        interrupt.resolved = true;
      }
    });
    
    return true;
  }
  
  addRecoveryHook(
    missionId: string,
    checkpointId: string,
    recoveryFunction: (checkpoint: MissionCheckpoint) => void,
    priority: number = 0
  ): void {
    const hooks = this.recoveryHooks.get(missionId) || [];
    
    hooks.push({
      checkpointId,
      recoveryFunction,
      priority
    });
    
    this.recoveryHooks.set(missionId, hooks);
  }
  
  getMission(missionId: string): Mission | null {
    return this.activeMissions.get(missionId) || null;
  }
  
  getActiveMissions(): Mission[] {
    return Array.from(this.activeMissions.values());
  }
  
  getMissionsByScroll(scrollReference: number): Mission[] {
    const missions: Mission[] = [];
    
    for (const mission of this.activeMissions.values()) {
      if (mission.scrollOrigin.includes(scrollReference)) {
        missions.push(mission);
      }
    }
    
    return missions;
  }
  
  private findRecoveryPath(mission: Mission): string | null {
    // Simple recovery path finding - can be made more sophisticated
    if (mission.checkpoints.length > 0) {
      const lastCheckpoint = mission.checkpoints[mission.checkpoints.length - 1];
      return `Recover from checkpoint ${lastCheckpoint.id} (${new Date(lastCheckpoint.timestamp).toISOString()})`;
    }
    
    return null;
  }
  
  private generateMissionId(): string {
    return `mission_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }
  
  private startPersistenceProcess(): void {
    // Periodic persistence of active missions
    setInterval(() => {
      for (const [missionId, mission] of this.activeMissions) {
        if (this.memoryArchitecture) {
          this.memoryArchitecture.store(
            mission,
            36,
            'mission',
            `mission_${missionId}_periodic`,
            MemoryScope.PERSISTENT
          );
        }
      }
    }, 30000); // Every 30 seconds
  }
  
  // Integration with value system
  evaluateMissionAlignment(missionId: string): number {
    const mission = this.activeMissions.get(missionId);
    if (!mission || !this.valueSystem) return 0;
    
    // Evaluate how well the mission aligns with embedded values
    const action = {
      type: 'pursue_mission',
      target: mission.rootGoal,
      parameters: mission.metadata
    };
    
    const alignmentScore = this.valueSystem.evaluateAction(action);
    return alignmentScore.score;
  }
  
  // Audit trail for scroll compliance
  getScrollTrace(missionId: string): Array<{
    scrollId: number;
    relevance: string;
    timestamp: number;
  }> {
    const mission = this.activeMissions.get(missionId) || 
                   this.missionHistory.find(m => m.id === missionId);
    
    if (!mission) return [];
    
    const trace = mission.scrollOrigin.map(scrollId => ({
      scrollId,
      relevance: this.getScrollRelevance(scrollId, mission),
      timestamp: mission.createdAt
    }));
    
    return trace;
  }
  
  private getScrollRelevance(scrollId: number, mission: Mission): string {
    // Map scroll IDs to their relevance for the mission
    const scrollRelevanceMap: { [key: number]: string } = {
      28: 'Agent Loops - Core execution structure',
      29: 'Objective Trees - Goal decomposition',
      30: 'Multi-Scale Goals - Temporal layering',
      36: 'Mission Memory - Persistence mechanism',
      // Add more mappings as needed
    };
    
    return scrollRelevanceMap[scrollId] || 'General guidance';
  }
  
  getStatistics() {
    const activeMissionCount = this.activeMissions.size;
    const completedMissions = this.missionHistory.filter(m => m.status === MissionStatus.COMPLETED).length;
    const failedMissions = this.missionHistory.filter(m => m.status === MissionStatus.FAILED).length;
    const interruptedMissions = Array.from(this.activeMissions.values())
      .filter(m => m.status === MissionStatus.INTERRUPTED).length;
    
    const totalCheckpoints = Array.from(this.activeMissions.values())
      .reduce((sum, mission) => sum + mission.checkpoints.length, 0);
    
    const avgProgress = Array.from(this.activeMissions.values())
      .filter(m => m.checkpoints.length > 0)
      .map(m => m.checkpoints[m.checkpoints.length - 1].progress)
      .reduce((sum, progress, _, arr) => sum + progress / arr.length, 0);
    
    return {
      activeMissionCount,
      completedMissions,
      failedMissions,
      interruptedMissions,
      totalCheckpoints,
      avgProgress,
      totalMissionsCreated: this.activeMissions.size + this.missionHistory.length
    };
  }
  
  // Export/import for persistence
  export(): string {
    const data = {
      activeMissions: Array.from(this.activeMissions.entries()),
      missionHistory: this.missionHistory,
      recoveryHooks: Array.from(this.recoveryHooks.entries()).map(([missionId, hooks]) => ({
        missionId,
        hooks: hooks.map(h => ({
          checkpointId: h.checkpointId,
          priority: h.priority
          // Note: recoveryFunction cannot be serialized
        }))
      }))
    };
    
    return JSON.stringify(data);
  }
  
  import(data: string): void {
    try {
      const parsed = JSON.parse(data);
      
      // Restore active missions
      this.activeMissions.clear();
      for (const [id, mission] of parsed.activeMissions) {
        this.activeMissions.set(id, mission);
      }
      
      // Restore mission history
      this.missionHistory = parsed.missionHistory;
      
      // Note: Recovery hooks with functions cannot be fully restored
      console.log('Mission memory imported. Note: Recovery functions must be re-registered.');
    } catch (error) {
      console.error('Failed to import mission memory:', error);
    }
  }
}
