// src/canon/core/MemoryArchitecture.ts
// Scroll #065: Memory Architecture - Structured knowledge systems

import { CognitiveMesh } from './CognitiveMesh';
import { AttentionSystem } from './AttentionSystem';

export interface MemoryTrace {
  id: string;
  content: any;
  timestamp: number;
  scrollReference?: number;
  role?: string;
  purpose?: string;
  strength: number;
  accessCount: number;
  lastAccessed: number;
}

export interface MemoryIndex {
  byScroll: Map<number, string[]>;
  byRole: Map<string, string[]>;
  byPurpose: Map<string, string[]>;
  byTime: Array<{timestamp: number, id: string}>;
}

export enum MemoryScope {
  IMMEDIATE = 'immediate',
  SESSIONAL = 'sessional',
  PERSISTENT = 'persistent',
  LEGACY = 'legacy'
}

export class MemoryArchitecture {
  private memories: Map<string, MemoryTrace> = new Map();
  private index: MemoryIndex = {
    byScroll: new Map(),
    byRole: new Map(),
    byPurpose: new Map(),
    byTime: []
  };
  
  private forgettingThreshold: number = 0.1;
  private compressionThreshold: number = 100;
  private maxMemories: number = 10000;
  
  constructor(
    private cognitiveMesh?: CognitiveMesh,
    private attentionSystem?: AttentionSystem
  ) {
    this.startMaintenanceProcess();
  }
  
  store(
    content: any,
    scrollReference?: number,
    role?: string,
    purpose?: string,
    scope: MemoryScope = MemoryScope.SESSIONAL
  ): string {
    const id = this.generateMemoryId();
    const timestamp = Date.now();
    
    const memory: MemoryTrace = {
      id,
      content,
      timestamp,
      scrollReference,
      role,
      purpose,
      strength: 1.0,
      accessCount: 0,
      lastAccessed: timestamp
    };
    
    this.memories.set(id, memory);
    
    // Update indices
    if (scrollReference !== undefined) {
      const scrollMemories = this.index.byScroll.get(scrollReference) || [];
      scrollMemories.push(id);
      this.index.byScroll.set(scrollReference, scrollMemories);
    }
    
    if (role) {
      const roleMemories = this.index.byRole.get(role) || [];
      roleMemories.push(id);
      this.index.byRole.set(role, roleMemories);
    }
    
    if (purpose) {
      const purposeMemories = this.index.byPurpose.get(purpose) || [];
      purposeMemories.push(id);
      this.index.byPurpose.set(purpose, purposeMemories);
    }
    
    this.index.byTime.push({ timestamp, id });
    
    // Update cognitive mesh if available
    if (this.cognitiveMesh && purpose) {
      this.cognitiveMesh.addNode(`memory:${id}`, purpose, { memory: true, scope });
    }
    
    // Trigger compression if needed
    if (this.memories.size > this.compressionThreshold) {
      this.compress();
    }
    
    return id;
  }
  
  retrieve(id: string): MemoryTrace | null {
    const memory = this.memories.get(id);
    if (!memory) return null;
    
    // Update access metadata
    memory.accessCount++;
    memory.lastAccessed = Date.now();
    memory.strength = Math.min(1.0, memory.strength * 1.1);
    
    // Update attention if available
    if (this.attentionSystem) {
      this.attentionSystem.focus(`memory:${id}`, memory.strength, 3000, {
        memoryAccess: true,
        purpose: memory.purpose
      });
    }
    
    return memory;
  }
  
  query(criteria: {
    scrollReference?: number;
    role?: string;
    purpose?: string;
    timeRange?: { start: number; end: number };
    minStrength?: number;
  }): MemoryTrace[] {
    let candidates = new Set<string>();
    
    // Start with all memories if no specific criteria
    if (!criteria.scrollReference && !criteria.role && !criteria.purpose) {
      candidates = new Set(this.memories.keys());
    }
    
    // Filter by scroll reference
    if (criteria.scrollReference !== undefined) {
      const scrollMemories = this.index.byScroll.get(criteria.scrollReference) || [];
      for (const id of scrollMemories) {
        candidates.add(id);
      }
    }
    
    // Filter by role
    if (criteria.role) {
      const roleMemories = this.index.byRole.get(criteria.role) || [];
      if (candidates.size === 0) {
        roleMemories.forEach(id => candidates.add(id));
      } else {
        candidates = new Set(roleMemories.filter(id => candidates.has(id)));
      }
    }
    
    // Filter by purpose
    if (criteria.purpose) {
      const purposeMemories = this.index.byPurpose.get(criteria.purpose) || [];
      if (candidates.size === 0) {
        purposeMemories.forEach(id => candidates.add(id));
      } else {
        candidates = new Set(purposeMemories.filter(id => candidates.has(id)));
      }
    }
    
    // Convert to memory traces and apply remaining filters
    const results: MemoryTrace[] = [];
    
    for (const id of candidates) {
      const memory = this.memories.get(id);
      if (!memory) continue;
      
      // Time range filter
      if (criteria.timeRange) {
        if (memory.timestamp < criteria.timeRange.start || 
            memory.timestamp > criteria.timeRange.end) {
          continue;
        }
      }
      
      // Strength filter
      if (criteria.minStrength !== undefined && memory.strength < criteria.minStrength) {
        continue;
      }
      
      results.push(memory);
    }
    
    // Sort by relevance (strength * recency)
    return results.sort((a, b) => {
      const scoreA = a.strength * (1 / (Date.now() - a.lastAccessed + 1));
      const scoreB = b.strength * (1 / (Date.now() - b.lastAccessed + 1));
      return scoreB - scoreA;
    });
  }
  
  forget(id: string, reason?: string): boolean {
    const memory = this.memories.get(id);
    if (!memory) return false;
    
    // Log forgetting event
    console.log(`Forgetting memory ${id}: ${reason || 'No reason specified'}`);
    
    // Remove from main storage
    this.memories.delete(id);
    
    // Remove from indices
    if (memory.scrollReference !== undefined) {
      const scrollMemories = this.index.byScroll.get(memory.scrollReference) || [];
      this.index.byScroll.set(
        memory.scrollReference,
        scrollMemories.filter(mid => mid !== id)
      );
    }
    
    if (memory.role) {
      const roleMemories = this.index.byRole.get(memory.role) || [];
      this.index.byRole.set(
        memory.role,
        roleMemories.filter(mid => mid !== id)
      );
    }
    
    if (memory.purpose) {
      const purposeMemories = this.index.byPurpose.get(memory.purpose) || [];
      this.index.byPurpose.set(
        memory.purpose,
        purposeMemories.filter(mid => mid !== id)
      );
    }
    
    this.index.byTime = this.index.byTime.filter(entry => entry.id !== id);
    
    return true;
  }
  
  compress(): void {
    // Identify memories to compress or forget
    const allMemories = Array.from(this.memories.values());
    
    // Sort by importance (combination of strength, access count, and recency)
    allMemories.sort((a, b) => {
      const scoreA = this.calculateImportance(a);
      const scoreB = this.calculateImportance(b);
      return scoreB - scoreA;
    });
    
    // Keep only the most important memories
    const toKeep = allMemories.slice(0, Math.floor(this.maxMemories * 0.8));
    const toCompress = allMemories.slice(
      Math.floor(this.maxMemories * 0.8),
      Math.floor(this.maxMemories * 0.95)
    );
    const toForget = allMemories.slice(Math.floor(this.maxMemories * 0.95));
    
    // Forget least important memories
    for (const memory of toForget) {
      this.forget(memory.id, 'Compression: Low importance');
    }
    
    // Compress medium importance memories
    for (const memory of toCompress) {
      this.compressMemory(memory);
    }
  }
  
  private compressMemory(memory: MemoryTrace): void {
    // Create compressed version
    const compressed = {
      summary: this.summarize(memory.content),
      originalId: memory.id,
      timestamp: memory.timestamp,
      importance: this.calculateImportance(memory)
    };
    
    // Replace content with compressed version
    memory.content = compressed;
    memory.strength *= 0.8; // Reduce strength for compressed memories
  }
  
  private calculateImportance(memory: MemoryTrace): number {
    const recencyFactor = 1 / (Date.now() - memory.lastAccessed + 1);
    const accessFactor = Math.log(memory.accessCount + 1);
    const strengthFactor = memory.strength;
    
    return recencyFactor * accessFactor * strengthFactor;
  }
  
  private summarize(content: any): string {
    // Simple summarization - in practice, this would be more sophisticated
    if (typeof content === 'string') {
      return content.substring(0, 100) + '...';
    } else if (typeof content === 'object') {
      return JSON.stringify(content).substring(0, 100) + '...';
    }
    return String(content).substring(0, 100) + '...';
  }
  
  private generateMemoryId(): string {
    return `mem_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }
  
  private startMaintenanceProcess(): void {
    // Periodic decay and cleanup
    setInterval(() => {
      for (const memory of this.memories.values()) {
        // Decay strength based on time since last access
        const timeSinceAccess = Date.now() - memory.lastAccessed;
        const decayFactor = Math.exp(-timeSinceAccess / (24 * 60 * 60 * 1000)); // 1 day half-life
        memory.strength *= decayFactor;
        
        // Mark for forgetting if below threshold
        if (memory.strength < this.forgettingThreshold) {
          this.forget(memory.id, 'Natural decay');
        }
      }
      
      // Compress if needed
      if (this.memories.size > this.maxMemories * 0.9) {
        this.compress();
      }
    }, 60000); // Run every minute
  }
  
  // Export/import for persistence
  export(): string {
    const data = {
      memories: Array.from(this.memories.entries()),
      index: {
        byScroll: Array.from(this.index.byScroll.entries()),
        byRole: Array.from(this.index.byRole.entries()),
        byPurpose: Array.from(this.index.byPurpose.entries()),
        byTime: this.index.byTime
      }
    };
    
    return JSON.stringify(data);
  }
  
  import(data: string): void {
    try {
      const parsed = JSON.parse(data);
      
      // Restore memories
      this.memories.clear();
      for (const [id, memory] of parsed.memories) {
        this.memories.set(id, memory);
      }
      
      // Restore indices
      this.index.byScroll = new Map(parsed.index.byScroll);
      this.index.byRole = new Map(parsed.index.byRole);
      this.index.byPurpose = new Map(parsed.index.byPurpose);
      this.index.byTime = parsed.index.byTime;
    } catch (error) {
      console.error('Failed to import memory architecture:', error);
    }
  }
  
  getStatistics() {
    const totalMemories = this.memories.size;
    const avgStrength = Array.from(this.memories.values())
      .reduce((sum, mem) => sum + mem.strength, 0) / totalMemories;
    
    const memoryByScope = new Map<string, number>();
    const memoryByRole = new Map<string, number>();
    
    for (const memory of this.memories.values()) {
      // Count by role
      if (memory.role) {
        memoryByRole.set(memory.role, (memoryByRole.get(memory.role) || 0) + 1);
      }
    }
    
    return {
      totalMemories,
      avgStrength,
      memoryByRole: Object.fromEntries(memoryByRole),
      oldestMemory: this.index.byTime[0]?.timestamp || null,
      newestMemory: this.index.byTime[this.index.byTime.length - 1]?.timestamp || null
    };
  }
}
