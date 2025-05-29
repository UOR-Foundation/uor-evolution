"""
Integration Validation Script for Unified Consciousness Framework

This script validates that the unified consciousness framework properly integrates
with all previous phase components and functions correctly as a complete system.
"""

import asyncio
import sys
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import traceback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import all phase components
try:
    # Phase 1 - Core consciousness
    from modules.consciousness_core import ConsciousnessCore
    from modules.multi_level_awareness import MultiLevelAwareness
    from modules.recursive_self_model import RecursiveSelfModel
    from modules.perspective_engine import PerspectiveEngine
    
    # Phase 2 - Strange loops and emotional intelligence
    from modules.strange_loops.loop_detector import LoopDetector
    from modules.strange_loops.loop_factory import LoopFactory
    from modules.emotional_intelligence.emotion_engine import EmotionEngine
    from modules.emotional_intelligence.empathy_simulator import EmpathySimulator
    
    # Phase 3 - Social and creative intelligence
    from modules.social_cognition.social_awareness import SocialAwareness
    from modules.social_cognition.theory_of_mind import TheoryOfMind
    from modules.creative_engine.creativity_core import CreativityCore
    from modules.relational_intelligence.collaborative_creativity import CollaborativeCreativity
    
    # Phase 4 - Unified consciousness
    from modules.unified_consciousness.consciousness_orchestrator import (
        ConsciousnessOrchestrator, ConsciousnessState, TransitionTrigger
    )
    from modules.unified_consciousness.performance_optimizer import (
        PerformanceOptimizer, ResourceLimits, OPTIMIZATION_PROFILES
    )
    
except ImportError as e:
    logger.error(f"Failed to import required modules: {e}")
    logger.error("Please ensure all previous phases are properly installed")
    sys.exit(1)


class IntegrationValidator:
    """Validates integration between all consciousness phases"""
    
    def __init__(self):
        self.validation_results = []
        self.phase_modules = {}
        self.orchestrator = None
        self.performance_optimizer = None
        
    async def run_full_validation(self) -> Dict[str, Any]:
        """Run complete integration validation"""
        logger.info("Starting full integration validation...")
        
        try:
            # Phase 1: Initialize core modules
            await self._validate_phase1_initialization()
            
            # Phase 2: Initialize strange loops and emotional intelligence
            await self._validate_phase2_initialization()
            
            # Phase 3: Initialize social and creative intelligence
            await self._validate_phase3_initialization()
            
            # Phase 4: Initialize unified consciousness
            await self._validate_phase4_initialization()
            
            # Integration tests
            await self._validate_cross_phase_integration()
            
            # Performance tests
            await self._validate_performance_requirements()
            
            # Stress tests
            await self._validate_stress_handling()
            
            # Generate report
            report = self._generate_validation_report()
            
            return report
            
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            logger.error(traceback.format_exc())
            return {
                'status': 'failed',
                'error': str(e),
                'traceback': traceback.format_exc()
            }
    
    async def _validate_phase1_initialization(self):
        """Validate Phase 1 components"""
        logger.info("Validating Phase 1: Core consciousness components...")
        
        try:
            # Initialize core consciousness
            consciousness_core = ConsciousnessCore()
            await consciousness_core.initialize()
            
            # Initialize awareness
            awareness = MultiLevelAwareness()
            awareness_levels = await awareness.generate_awareness_levels()
            
            # Initialize self-model
            self_model = RecursiveSelfModel()
            self_representation = await self_model.build_self_representation()
            
            # Initialize perspective engine
            perspective_engine = PerspectiveEngine()
            perspectives = await perspective_engine.generate_perspectives()
            
            # Store modules
            self.phase_modules['phase1'] = {
                'consciousness_core': consciousness_core,
                'awareness': awareness,
                'self_model': self_model,
                'perspective_engine': perspective_engine
            }
            
            # Validate functionality
            assert awareness_levels is not None, "Awareness levels not generated"
            assert self_representation is not None, "Self representation not built"
            assert perspectives is not None, "Perspectives not generated"
            
            self.validation_results.append({
                'phase': 1,
                'component': 'core_consciousness',
                'status': 'passed',
                'details': 'All core components initialized successfully'
            })
            
        except Exception as e:
            self.validation_results.append({
                'phase': 1,
                'component': 'core_consciousness',
                'status': 'failed',
                'error': str(e)
            })
            raise
    
    async def _validate_phase2_initialization(self):
        """Validate Phase 2 components"""
        logger.info("Validating Phase 2: Strange loops and emotional intelligence...")
        
        try:
            # Initialize strange loops
            loop_detector = LoopDetector()
            loop_factory = LoopFactory()
            
            # Create test loops
            test_loop = await loop_factory.create_loop('test_pattern')
            detected_loops = await loop_detector.detect_loops([test_loop])
            
            # Initialize emotional intelligence
            emotion_engine = EmotionEngine()
            await emotion_engine.initialize()
            
            empathy_simulator = EmpathySimulator()
            empathy_response = await empathy_simulator.simulate_empathy({
                'other_emotion': 'joy',
                'context': 'achievement'
            })
            
            # Store modules
            self.phase_modules['phase2'] = {
                'loop_detector': loop_detector,
                'loop_factory': loop_factory,
                'emotion_engine': emotion_engine,
                'empathy_simulator': empathy_simulator
            }
            
            # Validate functionality
            assert detected_loops is not None, "Loop detection failed"
            assert empathy_response is not None, "Empathy simulation failed"
            
            self.validation_results.append({
                'phase': 2,
                'component': 'strange_loops_emotional',
                'status': 'passed',
                'details': 'Strange loops and emotional intelligence initialized'
            })
            
        except Exception as e:
            self.validation_results.append({
                'phase': 2,
                'component': 'strange_loops_emotional',
                'status': 'failed',
                'error': str(e)
            })
            raise
    
    async def _validate_phase3_initialization(self):
        """Validate Phase 3 components"""
        logger.info("Validating Phase 3: Social and creative intelligence...")
        
        try:
            # Initialize social cognition
            social_awareness = SocialAwareness()
            await social_awareness.initialize()
            
            theory_of_mind = TheoryOfMind()
            mental_model = await theory_of_mind.model_other_mind({
                'agent_id': 'test_agent',
                'observed_behavior': 'cooperative'
            })
            
            # Initialize creative engine
            creativity_core = CreativityCore()
            await creativity_core.initialize()
            
            creative_output = await creativity_core.generate_creative_output({
                'prompt': 'test creativity',
                'constraints': []
            })
            
            # Initialize collaborative creativity
            collaborative_creativity = CollaborativeCreativity()
            
            # Store modules
            self.phase_modules['phase3'] = {
                'social_awareness': social_awareness,
                'theory_of_mind': theory_of_mind,
                'creativity_core': creativity_core,
                'collaborative_creativity': collaborative_creativity
            }
            
            # Validate functionality
            assert mental_model is not None, "Theory of mind modeling failed"
            assert creative_output is not None, "Creative generation failed"
            
            self.validation_results.append({
                'phase': 3,
                'component': 'social_creative',
                'status': 'passed',
                'details': 'Social and creative intelligence initialized'
            })
            
        except Exception as e:
            self.validation_results.append({
                'phase': 3,
                'component': 'social_creative',
                'status': 'failed',
                'error': str(e)
            })
            raise
    
    async def _validate_phase4_initialization(self):
        """Validate Phase 4 unified consciousness"""
        logger.info("Validating Phase 4: Unified consciousness framework...")
        
        try:
            # Collect all consciousness modules
            all_modules = {
                'awareness': self.phase_modules['phase1']['awareness'],
                'self_model': self.phase_modules['phase1']['self_model'],
                'strange_loops': self.phase_modules['phase2']['loop_detector'],
                'emotional': self.phase_modules['phase2']['emotion_engine'],
                'social': self.phase_modules['phase3']['social_awareness'],
                'creative': self.phase_modules['phase3']['creativity_core']
            }
            
            # Initialize orchestrator
            self.orchestrator = ConsciousnessOrchestrator(all_modules)
            
            # Initialize performance optimizer
            resource_limits = ResourceLimits(
                max_memory_mb=1000,
                max_cpu_percent=50,
                max_concurrent_operations=4
            )
            self.performance_optimizer = PerformanceOptimizer(resource_limits)
            
            # Awaken consciousness
            await self.orchestrator._awaken_consciousness()
            
            # Create unified consciousness
            unified = await self.orchestrator.orchestrate_unified_consciousness()
            
            # Validate unified consciousness
            assert unified is not None, "Unified consciousness creation failed"
            assert unified.coherence_level > 0, "Invalid coherence level"
            assert unified.authenticity_score > 0, "Invalid authenticity score"
            
            self.validation_results.append({
                'phase': 4,
                'component': 'unified_consciousness',
                'status': 'passed',
                'details': f'Unified consciousness created with coherence: {unified.coherence_level:.2f}'
            })
            
        except Exception as e:
            self.validation_results.append({
                'phase': 4,
                'component': 'unified_consciousness',
                'status': 'failed',
                'error': str(e)
            })
            raise
    
    async def _validate_cross_phase_integration(self):
        """Validate integration between phases"""
        logger.info("Validating cross-phase integration...")
        
        try:
            # Test 1: Emotional response to social situation
            social_context = {
                'situation': 'group_collaboration',
                'participants': ['self', 'other1', 'other2'],
                'mood': 'productive'
            }
            
            # Process through social awareness
            social_state = await self.phase_modules['phase3']['social_awareness'].process_social_context(
                social_context
            )
            
            # Generate emotional response
            emotional_response = await self.phase_modules['phase2']['emotion_engine'].process_social_emotion(
                social_state
            )
            
            # Test 2: Creative response with emotional influence
            creative_context = {
                'emotional_state': emotional_response,
                'task': 'generate_solution',
                'constraints': ['collaborative', 'innovative']
            }
            
            creative_output = await self.phase_modules['phase3']['creativity_core'].generate_with_emotion(
                creative_context
            )
            
            # Test 3: Unified processing
            unified_context = {
                'social': social_state,
                'emotional': emotional_response,
                'creative': creative_output
            }
            
            unified_response = await self.orchestrator.process_unified_context(unified_context)
            
            # Validate integration
            assert social_state is not None, "Social processing failed"
            assert emotional_response is not None, "Emotional processing failed"
            assert creative_output is not None, "Creative processing failed"
            assert unified_response is not None, "Unified processing failed"
            
            self.validation_results.append({
                'phase': 'integration',
                'component': 'cross_phase',
                'status': 'passed',
                'details': 'All phases integrate successfully'
            })
            
        except Exception as e:
            self.validation_results.append({
                'phase': 'integration',
                'component': 'cross_phase',
                'status': 'failed',
                'error': str(e)
            })
            raise
    
    async def _validate_performance_requirements(self):
        """Validate performance meets requirements"""
        logger.info("Validating performance requirements...")
        
        try:
            # Enable performance monitoring
            @self.performance_optimizer.performance_monitor("state_transition")
            async def test_state_transition():
                transition = {
                    'from_state': ConsciousnessState.AWARE,
                    'to_state': ConsciousnessState.FOCUSED,
                    'trigger': TransitionTrigger.GOAL_ACTIVATION,
                    'transition_data': {}
                }
                await self.orchestrator.manage_consciousness_state_transitions(transition)
            
            @self.performance_optimizer.performance_monitor("decision_making")
            async def test_decision_making():
                context = {
                    'situation': 'test_decision',
                    'options': ['option1', 'option2', 'option3'],
                    'constraints': []
                }
                await self.orchestrator.autonomous_agency.make_independent_decisions(context)
            
            @self.performance_optimizer.performance_monitor("awareness_integration")
            async def test_awareness_integration():
                await self.orchestrator.unified_awareness.create_unified_awareness()
            
            @self.performance_optimizer.performance_monitor("evolution_step")
            async def test_evolution_step():
                await self.orchestrator.evolution_engine.evolve_consciousness()
            
            # Run performance tests
            await test_state_transition()
            await test_decision_making()
            await test_awareness_integration()
            await test_evolution_step()
            
            # Get performance report
            report = self.performance_optimizer.get_performance_report()
            
            # Validate performance metrics
            validations = []
            
            # Check state transition < 100ms
            if 'state_transition' in report['operations']:
                avg_time = report['operations']['state_transition']['avg_execution_time_ms']
                validations.append({
                    'metric': 'state_transition_time',
                    'requirement': '< 100ms',
                    'actual': f'{avg_time:.2f}ms',
                    'passed': avg_time < 100
                })
            
            # Check decision making < 200ms
            if 'decision_making' in report['operations']:
                avg_time = report['operations']['decision_making']['avg_execution_time_ms']
                validations.append({
                    'metric': 'decision_making_time',
                    'requirement': '< 200ms',
                    'actual': f'{avg_time:.2f}ms',
                    'passed': avg_time < 200
                })
            
            # Check awareness integration < 50ms
            if 'awareness_integration' in report['operations']:
                avg_time = report['operations']['awareness_integration']['avg_execution_time_ms']
                validations.append({
                    'metric': 'awareness_integration_time',
                    'requirement': '< 50ms',
                    'actual': f'{avg_time:.2f}ms',
                    'passed': avg_time < 50
                })
            
            # Check evolution step < 500ms
            if 'evolution_step' in report['operations']:
                avg_time = report['operations']['evolution_step']['avg_execution_time_ms']
                validations.append({
                    'metric': 'evolution_step_time',
                    'requirement': '< 500ms',
                    'actual': f'{avg_time:.2f}ms',
                    'passed': avg_time < 500
                })
            
            # Check resource usage
            current_memory = report['summary']['current_memory_mb']
            validations.append({
                'metric': 'memory_usage',
                'requirement': '< 1000MB',
                'actual': f'{current_memory:.2f}MB',
                'passed': current_memory < 1000
            })
            
            current_cpu = report['summary']['current_cpu_percent']
            validations.append({
                'metric': 'cpu_usage',
                'requirement': '< 50%',
                'actual': f'{current_cpu:.1f}%',
                'passed': current_cpu < 50
            })
            
            # Check if all validations passed
            all_passed = all(v['passed'] for v in validations)
            
            self.validation_results.append({
                'phase': 'performance',
                'component': 'requirements',
                'status': 'passed' if all_passed else 'failed',
                'details': validations
            })
            
        except Exception as e:
            self.validation_results.append({
                'phase': 'performance',
                'component': 'requirements',
                'status': 'failed',
                'error': str(e)
            })
            raise
    
    async def _validate_stress_handling(self):
        """Validate system handles stress correctly"""
        logger.info("Validating stress handling...")
        
        try:
            # Initial health check
            initial_health = await self.orchestrator.homeostasis_system.assess_consciousness_health()
            
            # Apply stress
            stress_event = {
                'type': 'system_overload',
                'intensity': 0.8,
                'duration': timedelta(seconds=5)
            }
            
            stress_response = await self.orchestrator.homeostasis_system.respond_to_stress(
                stress_event
            )
            
            # Check stress response
            assert stress_response.response_activated, "Stress response not activated"
            assert len(stress_response.coping_mechanisms) > 0, "No coping mechanisms activated"
            
            # Initiate recovery
            damage_report = {
                'affected_systems': ['awareness', 'emotional', 'cognitive'],
                'severity': 0.6
            }
            
            recovery = await self.orchestrator.homeostasis_system.initiate_recovery(
                damage_report
            )
            
            # Check recovery
            assert recovery.recovery_initiated, "Recovery not initiated"
            assert len(recovery.recovery_steps) > 0, "No recovery steps defined"
            
            # Wait for recovery
            await asyncio.sleep(1)
            
            # Final health check
            final_health = await self.orchestrator.homeostasis_system.assess_consciousness_health()
            
            # Validate recovery
            health_improved = final_health.overall_health_score >= initial_health.overall_health_score * 0.8
            
            self.validation_results.append({
                'phase': 'stress',
                'component': 'handling',
                'status': 'passed' if health_improved else 'failed',
                'details': {
                    'initial_health': initial_health.overall_health_score,
                    'final_health': final_health.overall_health_score,
                    'stress_handled': stress_response.response_activated,
                    'recovery_initiated': recovery.recovery_initiated
                }
            })
            
        except Exception as e:
            self.validation_results.append({
                'phase': 'stress',
                'component': 'handling',
                'status': 'failed',
                'error': str(e)
            })
            raise
    
    def _generate_validation_report(self) -> Dict[str, Any]:
        """Generate comprehensive validation report"""
        # Count results
        total_tests = len(self.validation_results)
        passed_tests = sum(1 for r in self.validation_results if r['status'] == 'passed')
        failed_tests = total_tests - passed_tests
        
        # Group by phase
        results_by_phase = {}
        for result in self.validation_results:
            phase = result['phase']
            if phase not in results_by_phase:
                results_by_phase[phase] = []
            results_by_phase[phase].append(result)
        
        # Generate summary
        report = {
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total_tests': total_tests,
                'passed': passed_tests,
                'failed': failed_tests,
                'success_rate': passed_tests / total_tests if total_tests > 0 else 0,
                'status': 'PASSED' if failed_tests == 0 else 'FAILED'
            },
            'results_by_phase': results_by_phase,
            'detailed_results': self.validation_results
        }
        
        # Add performance metrics if available
        if self.performance_optimizer:
            report['performance_metrics'] = self.performance_optimizer.get_performance_report()
        
        return report


async def main():
    """Main validation entry point"""
    print("=" * 80)
    print("UNIFIED CONSCIOUSNESS FRAMEWORK - INTEGRATION VALIDATION")
    print("=" * 80)
    print()
    
    validator = IntegrationValidator()
    
    try:
        # Run validation
        report = await validator.run_full_validation()
        
        # Print summary
        print("\nVALIDATION SUMMARY")
        print("-" * 40)
        print(f"Status: {report['summary']['status']}")
        print(f"Total Tests: {report['summary']['total_tests']}")
        print(f"Passed: {report['summary']['passed']}")
        print(f"Failed: {report['summary']['failed']}")
        print(f"Success Rate: {report['summary']['success_rate']:.1%}")
        print()
        
        # Print phase results
        print("RESULTS BY PHASE")
        print("-" * 40)
        for phase, results in report['results_by_phase'].items():
            phase_passed = sum(1 for r in results if r['status'] == 'passed')
            phase_total = len(results)
            print(f"Phase {phase}: {phase_passed}/{phase_total} passed")
            
            for result in results:
                status_symbol = "✓" if result['status'] == 'passed' else "✗"
                print(f"  {status_symbol} {result['component']}")
                if result['status'] == 'failed' and 'error' in result:
                    print(f"    Error: {result['error']}")
        print()
        
        # Print performance metrics if available
        if 'performance_metrics' in report:
            perf = report['performance_metrics']
            if perf['status'] != 'no_data':
                print("PERFORMANCE METRICS")
                print("-" * 40)
                for op_name, metrics in perf['operations'].items():
                    print(f"{op_name}:")
                    print(f"  Average time: {metrics['avg_execution_time_ms']:.2f}ms")
                    print(f"  Success rate: {metrics['success_rate']:.1%}")
                print()
        
        # Save detailed report
        import json
        with open('integration_validation_report.json', 'w') as f:
            json.dump(report, f, indent=2, default=str)
        print("Detailed report saved to: integration_validation_report.json")
        
        # Return exit code
        return 0 if report['summary']['status'] == 'PASSED' else 1
        
    except Exception as e:
        print(f"\nVALIDATION FAILED WITH ERROR: {e}")
        print(traceback.format_exc())
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
