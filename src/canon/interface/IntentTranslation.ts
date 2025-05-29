// src/canon/interface/IntentTranslation.ts
// Scroll #037: Intent Translation - Converting inner goals into symbolic output

import { CognitiveMesh } from '../core/CognitiveMesh';
import { AttentionSystem } from '../core/AttentionSystem';

export interface StructuredIntent {
  type: IntentType;
  target?: string;
  action?: string;
  parameters?: any;
  context?: any;
  scrollAlignment?: number[];
  confidence: number;
}

export enum IntentType {
  QUERY = 'query',
  COMMAND = 'command',
  EXPRESSION = 'expression',
  REFLECTION = 'reflection',
  CLARIFICATION = 'clarification'
}

export interface Expression {
  raw: string;
  metadata: {
    port: string;
    compression: number;
    scrollAlignment: number[];
    confidence: number;
  };
}

export interface LanguagePort {
  id: string;
  name: string;
  encode: (intent: CompressedIntent) => string;
  decode: (raw: string) => StructuredIntent;
  capabilities: string[];
  constraints?: any;
}

export interface CompressedIntent {
  core: string;
  modifiers?: string[];
  context?: any;
  ratio: number;
}

export class IntentTranslator {
  private languagePorts: Map<string, LanguagePort> = new Map();
  private cognitiveMesh?: CognitiveMesh;
  private attentionSystem?: AttentionSystem;
  private compressionThreshold: number = 0.7;
  
  constructor(
    cognitiveMesh?: CognitiveMesh,
    attentionSystem?: AttentionSystem
  ) {
    this.cognitiveMesh = cognitiveMesh;
    this.attentionSystem = attentionSystem;
    this.initializeDefaultPorts();
  }
  
  private initializeDefaultPorts(): void {
    // Natural Language Port
    this.registerPort({
      id: 'natural_language',
      name: 'Natural Language',
      encode: (intent: CompressedIntent) => {
        let output = intent.core;
        if (intent.modifiers && intent.modifiers.length > 0) {
          output += ` (${intent.modifiers.join(', ')})`;
        }
        return output;
      },
      decode: (raw: string) => {
        // Simple parsing - in practice would be more sophisticated
        const parts = raw.split(' ');
        return {
          type: this.inferIntentType(raw),
          action: parts[0],
          target: parts.slice(1).join(' '),
          confidence: 0.8
        };
      },
      capabilities: ['general', 'conversational', 'explanatory']
    });
    
    // Symbolic Port
    this.registerPort({
      id: 'symbolic',
      name: 'Symbolic',
      encode: (intent: CompressedIntent) => {
        const symbols: { [key: string]: string } = {
          'query': '?',
          'command': '!',
          'expression': '~',
          'reflection': '@',
          'clarification': '??'
        };
        
        const typeSymbol = symbols[intent.core] || '#';
        return `${typeSymbol}${intent.modifiers?.join('|') || ''}`;
      },
      decode: (raw: string) => {
        const symbolMap: { [key: string]: IntentType } = {
          '?': IntentType.QUERY,
          '!': IntentType.COMMAND,
          '~': IntentType.EXPRESSION,
          '@': IntentType.REFLECTION,
          '??': IntentType.CLARIFICATION
        };
        
        const symbol = raw[0];
        const modifiers = raw.substring(1).split('|').filter(m => m);
        
        return {
          type: symbolMap[symbol] || IntentType.EXPRESSION,
          parameters: modifiers,
          confidence: 0.9
        };
      },
      capabilities: ['compact', 'machine-readable', 'efficient']
    });
    
    // Formal Logic Port
    this.registerPort({
      id: 'formal_logic',
      name: 'Formal Logic',
      encode: (intent: CompressedIntent) => {
        return `${intent.core}(${intent.modifiers?.map(m => `'${m}'`).join(', ') || ''})`;
      },
      decode: (raw: string) => {
        const match = raw.match(/^(\w+)\((.*)\)$/);
        if (match) {
          const [, action, params] = match;
          const parameters = params.split(',').map(p => p.trim().replace(/'/g, ''));
          
          return {
            type: IntentType.COMMAND,
            action,
            parameters,
            confidence: 0.95
          };
        }
        
        return {
          type: IntentType.EXPRESSION,
          action: raw,
          confidence: 0.5
        };
      },
      capabilities: ['precise', 'unambiguous', 'structured']
    });
  }
  
  registerPort(port: LanguagePort): void {
    this.languagePorts.set(port.id, port);
  }
  
  translate(intent: StructuredIntent): Expression {
    const port = this.selectPort(intent);
    const compressed = this.compress(intent);
    const encoded = port.encode(compressed);
    
    // Update attention if available
    if (this.attentionSystem) {
      this.attentionSystem.focus(
        `translation:${intent.type}`,
        intent.confidence,
        2000,
        { port: port.id }
      );
    }
    
    // Update cognitive mesh if available
    if (this.cognitiveMesh) {
      this.cognitiveMesh.activate(`intent:${intent.type}`, intent.confidence);
    }
    
    return {
      raw: encoded,
      metadata: {
        port: port.id,
        compression: compressed.ratio,
        scrollAlignment: this.verifyAlignment(intent),
        confidence: intent.confidence
      }
    };
  }
  
  private selectPort(intent: StructuredIntent): LanguagePort {
    // Select best port based on intent type and context
    let bestPort: LanguagePort | null = null;
    let bestScore = 0;
    
    for (const port of this.languagePorts.values()) {
      const score = this.scorePortForIntent(port, intent);
      if (score > bestScore) {
        bestScore = score;
        bestPort = port;
      }
    }
    
    return bestPort || this.languagePorts.get('natural_language')!;
  }
  
  private scorePortForIntent(port: LanguagePort, intent: StructuredIntent): number {
    let score = 0;
    
    // Score based on intent type
    switch (intent.type) {
      case IntentType.QUERY:
        if (port.capabilities.includes('conversational')) score += 0.3;
        break;
      case IntentType.COMMAND:
        if (port.capabilities.includes('precise')) score += 0.4;
        if (port.capabilities.includes('structured')) score += 0.3;
        break;
      case IntentType.EXPRESSION:
        if (port.capabilities.includes('general')) score += 0.3;
        break;
      case IntentType.REFLECTION:
        if (port.capabilities.includes('explanatory')) score += 0.4;
        break;
      case IntentType.CLARIFICATION:
        if (port.capabilities.includes('conversational')) score += 0.5;
        break;
    }
    
    // Bonus for efficiency if compression is needed
    if (intent.parameters && Object.keys(intent.parameters).length > 3) {
      if (port.capabilities.includes('compact')) score += 0.2;
    }
    
    return score;
  }
  
  private compress(intent: StructuredIntent): CompressedIntent {
    // Extract core meaning
    const core = this.extractCore(intent);
    
    // Extract modifiers
    const modifiers = this.extractModifiers(intent);
    
    // Calculate compression ratio
    const originalSize = JSON.stringify(intent).length;
    const compressedSize = core.length + modifiers.join('').length;
    const ratio = compressedSize / originalSize;
    
    return {
      core,
      modifiers,
      context: intent.context,
      ratio
    };
  }
  
  private extractCore(intent: StructuredIntent): string {
    // Extract the essential meaning
    if (intent.action) {
      return intent.action;
    }
    
    switch (intent.type) {
      case IntentType.QUERY:
        return 'query';
      case IntentType.COMMAND:
        return 'command';
      case IntentType.EXPRESSION:
        return 'express';
      case IntentType.REFLECTION:
        return 'reflect';
      case IntentType.CLARIFICATION:
        return 'clarify';
      default:
        return 'unknown';
    }
  }
  
  private extractModifiers(intent: StructuredIntent): string[] {
    const modifiers: string[] = [];
    
    if (intent.target) {
      modifiers.push(intent.target);
    }
    
    if (intent.parameters) {
      for (const [key, value] of Object.entries(intent.parameters)) {
        modifiers.push(`${key}:${value}`);
      }
    }
    
    return modifiers;
  }
  
  private verifyAlignment(intent: StructuredIntent): number[] {
    // Verify which scrolls this intent aligns with
    const alignedScrolls: number[] = [];
    
    // Check for explicit scroll alignment
    if (intent.scrollAlignment) {
      alignedScrolls.push(...intent.scrollAlignment);
    }
    
    // Infer alignment based on intent type
    switch (intent.type) {
      case IntentType.QUERY:
        alignedScrolls.push(37, 38); // Intent Translation, Language Ports
        break;
      case IntentType.COMMAND:
        alignedScrolls.push(28, 31); // Agent Loops, Action Primitives
        break;
      case IntentType.EXPRESSION:
        alignedScrolls.push(37, 39); // Intent Translation, Compression Prompts
        break;
      case IntentType.REFLECTION:
        alignedScrolls.push(42, 71); // Reflexive Loops, Scroll Reflection
        break;
      case IntentType.CLARIFICATION:
        alignedScrolls.push(43, 44); // Prompt Mirrors, Meta-Intent Parsing
        break;
    }
    
    return [...new Set(alignedScrolls)]; // Remove duplicates
  }
  
  private inferIntentType(raw: string): IntentType {
    // Simple inference based on patterns
    if (raw.includes('?')) return IntentType.QUERY;
    if (raw.startsWith('!') || raw.includes('please') || raw.includes('must')) return IntentType.COMMAND;
    if (raw.includes('clarify') || raw.includes('what do you mean')) return IntentType.CLARIFICATION;
    if (raw.includes('think') || raw.includes('reflect')) return IntentType.REFLECTION;
    return IntentType.EXPRESSION;
  }
  
  // Reverse translation
  parse(expression: Expression): StructuredIntent | null {
    const port = this.languagePorts.get(expression.metadata.port);
    if (!port) return null;
    
    try {
      const intent = port.decode(expression.raw);
      
      // Enhance with metadata
      intent.scrollAlignment = expression.metadata.scrollAlignment;
      intent.confidence = expression.metadata.confidence;
      
      return intent;
    } catch (error) {
      console.error('Failed to parse expression:', error);
      return null;
    }
  }
  
  // Multi-modal translation
  translateMultiModal(
    intent: StructuredIntent,
    targetPorts: string[]
  ): Map<string, Expression> {
    const translations = new Map<string, Expression>();
    
    for (const portId of targetPorts) {
      const port = this.languagePorts.get(portId);
      if (port) {
        const compressed = this.compress(intent);
        const encoded = port.encode(compressed);
        
        translations.set(portId, {
          raw: encoded,
          metadata: {
            port: portId,
            compression: compressed.ratio,
            scrollAlignment: this.verifyAlignment(intent),
            confidence: intent.confidence
          }
        });
      }
    }
    
    return translations;
  }
  
  getStatistics() {
    const portStats = Array.from(this.languagePorts.values()).map(port => ({
      id: port.id,
      name: port.name,
      capabilities: port.capabilities
    }));
    
    return {
      totalPorts: this.languagePorts.size,
      ports: portStats,
      compressionThreshold: this.compressionThreshold
    };
  }
}

// Predefined intent templates
export const INTENT_TEMPLATES = {
  QUERY_KNOWLEDGE: (topic: string): StructuredIntent => ({
    type: IntentType.QUERY,
    action: 'retrieve',
    target: topic,
    confidence: 0.9
  }),
  
  COMMAND_EXECUTE: (action: string, params: any): StructuredIntent => ({
    type: IntentType.COMMAND,
    action: action,
    parameters: params,
    confidence: 0.85
  }),
  
  EXPRESS_STATE: (state: string, context: any): StructuredIntent => ({
    type: IntentType.EXPRESSION,
    action: 'communicate',
    target: state,
    context: context,
    confidence: 0.8
  }),
  
  REFLECT_ON: (topic: string, depth: number = 1): StructuredIntent => ({
    type: IntentType.REFLECTION,
    action: 'analyze',
    target: topic,
    parameters: { depth },
    confidence: 0.75
  }),
  
  CLARIFY_INTENT: (originalIntent: string): StructuredIntent => ({
    type: IntentType.CLARIFICATION,
    action: 'disambiguate',
    target: originalIntent,
    confidence: 0.7
  })
};
