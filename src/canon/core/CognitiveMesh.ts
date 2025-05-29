// src/canon/core/CognitiveMesh.ts
// Scroll #011: Cognitive Mesh - The dynamic web of thought nodes

import { NeuralPrimitive, PrimitiveType } from './NeuralPrimitives';
import { AttentionSystem } from './AttentionSystem';

export interface CognitiveNode {
  id: string;
  concept: string;
  activation: number;
  lastActivated: number;
  metadata: any;
  primitiveType?: PrimitiveType;
}

export interface CognitiveEdge {
  from: string;
  to: string;
  weight: number;
  type: EdgeType;
  metadata?: any;
}

export enum EdgeType {
  ASSOCIATION = 'association',
  CAUSATION = 'causation',
  SIMILARITY = 'similarity',
  CONTRAST = 'contrast',
  TEMPORAL = 'temporal',
  HIERARCHICAL = 'hierarchical'
}

export class CognitiveMesh {
  private nodes: Map<string, CognitiveNode> = new Map();
  private edges: Map<string, CognitiveEdge[]> = new Map();
  private activationThreshold: number = 0.1;
  private propagationDecay: number = 0.8;
  private maxPropagationDepth: number = 5;
  
  constructor(
    private attentionSystem?: AttentionSystem
  ) {}
  
  addNode(id: string, concept: string, metadata: any = {}, primitiveType?: PrimitiveType): void {
    this.nodes.set(id, {
      id,
      concept,
      activation: 0,
      lastActivated: Date.now(),
      metadata,
      primitiveType
    });
    
    if (!this.edges.has(id)) {
      this.edges.set(id, []);
    }
  }
  
  addEdge(from: string, to: string, weight: number, type: EdgeType, metadata?: any): void {
    const edge: CognitiveEdge = { from, to, weight, type, metadata };
    
    const fromEdges = this.edges.get(from) || [];
    fromEdges.push(edge);
    this.edges.set(from, fromEdges);
  }
  
  activate(nodeId: string, strength: number = 1.0): void {
    const node = this.nodes.get(nodeId);
    if (!node) return;
    
    // Update node activation
    node.activation = Math.min(1.0, node.activation + strength);
    node.lastActivated = Date.now();
    
    // Update attention system if available
    if (this.attentionSystem) {
      this.attentionSystem.focus(`mesh:${nodeId}`, strength, 5000, {
        concept: node.concept,
        meshActivation: true
      });
    }
    
    // Propagate activation
    this.propagateActivation(nodeId, strength, 0, new Set());
  }
  
  private propagateActivation(
    nodeId: string, 
    strength: number, 
    depth: number, 
    visited: Set<string>
  ): void {
    if (depth >= this.maxPropagationDepth || visited.has(nodeId)) {
      return;
    }
    
    visited.add(nodeId);
    const edges = this.edges.get(nodeId) || [];
    
    for (const edge of edges) {
      const propagatedStrength = strength * edge.weight * Math.pow(this.propagationDecay, depth);
      
      if (propagatedStrength > this.activationThreshold) {
        const targetNode = this.nodes.get(edge.to);
        if (targetNode) {
          targetNode.activation = Math.min(1.0, targetNode.activation + propagatedStrength);
          targetNode.lastActivated = Date.now();
          
          // Recursive propagation
          this.propagateActivation(edge.to, propagatedStrength, depth + 1, visited);
        }
      }
    }
  }
  
  findPath(fromId: string, toId: string, maxDepth: number = 10): string[] | null {
    const visited = new Set<string>();
    const queue: Array<{node: string, path: string[]}> = [{node: fromId, path: [fromId]}];
    
    while (queue.length > 0) {
      const current = queue.shift()!;
      
      if (current.path.length > maxDepth) {
        continue;
      }
      
      if (current.node === toId) {
        return current.path;
      }
      
      if (visited.has(current.node)) {
        continue;
      }
      
      visited.add(current.node);
      const edges = this.edges.get(current.node) || [];
      
      for (const edge of edges) {
        if (!visited.has(edge.to)) {
          queue.push({
            node: edge.to,
            path: [...current.path, edge.to]
          });
        }
      }
    }
    
    return null;
  }
  
  getActiveNodes(threshold: number = 0.1): CognitiveNode[] {
    return Array.from(this.nodes.values())
      .filter(node => node.activation > threshold)
      .sort((a, b) => b.activation - a.activation);
  }
  
  getRelatedConcepts(nodeId: string, depth: number = 2): Map<string, number> {
    const related = new Map<string, number>();
    const visited = new Set<string>();
    
    this.exploreRelated(nodeId, 1.0, depth, visited, related);
    
    // Remove the original node
    related.delete(nodeId);
    
    return related;
  }
  
  private exploreRelated(
    nodeId: string,
    strength: number,
    remainingDepth: number,
    visited: Set<string>,
    related: Map<string, number>
  ): void {
    if (remainingDepth <= 0 || visited.has(nodeId)) {
      return;
    }
    
    visited.add(nodeId);
    const current = related.get(nodeId) || 0;
    related.set(nodeId, current + strength);
    
    const edges = this.edges.get(nodeId) || [];
    for (const edge of edges) {
      const propagatedStrength = strength * edge.weight;
      this.exploreRelated(edge.to, propagatedStrength, remainingDepth - 1, visited, related);
    }
  }
  
  // Pattern completion - fill in missing connections
  completePattern(partialPattern: string[]): string[] {
    if (partialPattern.length < 2) return partialPattern;
    
    const completed = [...partialPattern];
    
    for (let i = 0; i < partialPattern.length - 1; i++) {
      const from = partialPattern[i];
      const to = partialPattern[i + 1];
      
      // Check if direct connection exists
      const edges = this.edges.get(from) || [];
      const directConnection = edges.find(e => e.to === to);
      
      if (!directConnection) {
        // Try to find intermediate nodes
        const path = this.findPath(from, to, 3);
        if (path && path.length > 2) {
          // Insert intermediate nodes
          completed.splice(i + 1, 0, ...path.slice(1, -1));
        }
      }
    }
    
    return completed;
  }
  
  // Lateral thinking - find unexpected connections
  findLateralConnections(nodeId: string, minSurprise: number = 0.3): Array<{node: string, surprise: number}> {
    const node = this.nodes.get(nodeId);
    if (!node) return [];
    
    const directConnections = new Set<string>();
    const edges = this.edges.get(nodeId) || [];
    
    for (const edge of edges) {
      directConnections.add(edge.to);
    }
    
    const lateralConnections: Array<{node: string, surprise: number}> = [];
    
    // Look for nodes that share connections but aren't directly connected
    for (const [otherId, otherNode] of this.nodes) {
      if (otherId === nodeId || directConnections.has(otherId)) {
        continue;
      }
      
      const otherEdges = this.edges.get(otherId) || [];
      const sharedConnections = otherEdges.filter(e => directConnections.has(e.to)).length;
      
      if (sharedConnections > 0) {
        const surprise = 1 / (sharedConnections + 1); // More shared = less surprise
        if (surprise >= minSurprise) {
          lateralConnections.push({ node: otherId, surprise });
        }
      }
    }
    
    return lateralConnections.sort((a, b) => b.surprise - a.surprise);
  }
  
  // Mesh rewiring based on activation patterns
  rewire(learningRate: number = 0.1): void {
    const activeNodes = this.getActiveNodes();
    
    // Strengthen connections between co-activated nodes
    for (let i = 0; i < activeNodes.length; i++) {
      for (let j = i + 1; j < activeNodes.length; j++) {
        const nodeA = activeNodes[i];
        const nodeB = activeNodes[j];
        
        // Check if connection exists
        const edges = this.edges.get(nodeA.id) || [];
        const existingEdge = edges.find(e => e.to === nodeB.id);
        
        if (existingEdge) {
          // Strengthen existing connection
          existingEdge.weight = Math.min(1.0, existingEdge.weight + learningRate * nodeA.activation * nodeB.activation);
        } else if (nodeA.activation * nodeB.activation > 0.5) {
          // Create new connection if co-activation is strong
          this.addEdge(nodeA.id, nodeB.id, learningRate, EdgeType.ASSOCIATION, {
            learned: true,
            timestamp: Date.now()
          });
        }
      }
    }
    
    // Decay unused connections
    for (const [nodeId, edges] of this.edges) {
      const node = this.nodes.get(nodeId);
      if (!node) continue;
      
      for (const edge of edges) {
        const targetNode = this.nodes.get(edge.to);
        if (targetNode && node.activation < 0.1 && targetNode.activation < 0.1) {
          edge.weight *= 0.95; // Decay
        }
      }
    }
  }
  
  // Get mesh statistics
  getStatistics() {
    const nodeCount = this.nodes.size;
    const edgeCount = Array.from(this.edges.values()).reduce((sum, edges) => sum + edges.length, 0);
    const activeNodes = this.getActiveNodes().length;
    
    // Calculate average connectivity
    const avgConnectivity = nodeCount > 0 ? edgeCount / nodeCount : 0;
    
    // Calculate clustering coefficient
    let clusteringSum = 0;
    let validNodes = 0;
    
    for (const [nodeId, edges] of this.edges) {
      if (edges.length >= 2) {
        const neighbors = edges.map(e => e.to);
        let triangles = 0;
        
        for (let i = 0; i < neighbors.length; i++) {
          for (let j = i + 1; j < neighbors.length; j++) {
            const neighborEdges = this.edges.get(neighbors[i]) || [];
            if (neighborEdges.some(e => e.to === neighbors[j])) {
              triangles++;
            }
          }
        }
        
        const possibleTriangles = (neighbors.length * (neighbors.length - 1)) / 2;
        if (possibleTriangles > 0) {
          clusteringSum += triangles / possibleTriangles;
          validNodes++;
        }
      }
    }
    
    const clusteringCoefficient = validNodes > 0 ? clusteringSum / validNodes : 0;
    
    return {
      nodeCount,
      edgeCount,
      activeNodes,
      avgConnectivity,
      clusteringCoefficient
    };
  }
}
