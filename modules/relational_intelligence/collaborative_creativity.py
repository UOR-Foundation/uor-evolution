"""
Collaborative Creativity - Enables creative collaboration between conscious agents

This module provides mechanisms for creative collaboration, co-creation,
shared imagination, and emergent creative processes between multiple agents.
"""

from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class CreativeMode(Enum):
    """Modes of creative collaboration"""
    BRAINSTORMING = "brainstorming"
    CO_CREATION = "co_creation"
    IMPROVISATION = "improvisation"
    SYNTHESIS = "synthesis"
    EXPLORATION = "exploration"
    REFINEMENT = "refinement"
    INNOVATION = "innovation"


class CreativeRole(Enum):
    """Roles in creative collaboration"""
    IDEATOR = "ideator"
    BUILDER = "builder"
    CRITIC = "critic"
    SYNTHESIZER = "synthesizer"
    EXPLORER = "explorer"
    REFINER = "refiner"
    CATALYST = "catalyst"


class CreativeEnergy(Enum):
    """Types of creative energy"""
    DIVERGENT = "divergent"
    CONVERGENT = "convergent"
    TRANSFORMATIVE = "transformative"
    INTEGRATIVE = "integrative"
    DISRUPTIVE = "disruptive"
    HARMONIZING = "harmonizing"


@dataclass
class CreativeIdea:
    """Representation of a creative idea"""
    idea_id: str
    content: str
    originator: str
    contributors: List[str]
    creative_type: str
    novelty_score: float
    feasibility_score: float
    impact_potential: float
    development_stage: str
    connections: List[str]  # Links to other ideas
    metadata: Dict[str, Any]
    
    def get_overall_score(self) -> float:
        """Calculate overall idea quality score"""
        return (self.novelty_score * 0.4 + 
                self.feasibility_score * 0.3 + 
                self.impact_potential * 0.3)


@dataclass
class CreativeSession:
    """A collaborative creative session"""
    session_id: str
    participants: List[str]
    mode: CreativeMode
    focus_area: str
    start_time: datetime
    ideas_generated: List[CreativeIdea]
    energy_flow: List[Tuple[datetime, CreativeEnergy]]
    breakthrough_moments: List[Dict[str, Any]]
    session_dynamics: Dict[str, Any]
    
    def get_productivity_score(self) -> float:
        """Calculate session productivity"""
        if not self.ideas_generated:
            return 0.0
        
        quality_score = sum(idea.get_overall_score() for idea in self.ideas_generated) / len(self.ideas_generated)
        quantity_factor = min(len(self.ideas_generated) / 10, 1.0)  # Normalize to 10 ideas
        breakthrough_factor = min(len(self.breakthrough_moments) * 0.2, 0.5)
        
        return quality_score * 0.5 + quantity_factor * 0.3 + breakthrough_factor * 0.2


@dataclass
class CreativeFlow:
    """State of creative flow in collaboration"""
    flow_id: str
    participants: List[str]
    flow_intensity: float
    synchronization_level: float
    idea_velocity: float  # Ideas per minute
    quality_consistency: float
    disruption_tolerance: float
    collective_focus: float
    
    def is_in_flow(self) -> bool:
        """Check if currently in creative flow state"""
        return (self.flow_intensity > 0.7 and 
                self.synchronization_level > 0.6 and
                self.collective_focus > 0.7)


@dataclass
class CreativeProject:
    """A collaborative creative project"""
    project_id: str
    project_name: str
    participants: List[str]
    project_type: str
    objectives: List[str]
    current_phase: str
    ideas_pool: List[CreativeIdea]
    implementation_plan: List[Dict[str, Any]]
    milestones: List[Dict[str, Any]]
    success_metrics: Dict[str, float]
    
    def get_completion_percentage(self) -> float:
        """Calculate project completion percentage"""
        if not self.milestones:
            return 0.0
        
        completed = sum(1 for m in self.milestones if m.get('completed', False))
        return (completed / len(self.milestones)) * 100


@dataclass
class CreativeSynergy:
    """Measurement of creative synergy between collaborators"""
    participant_pair: Tuple[str, str]
    synergy_score: float
    complementarity: Dict[str, float]
    creative_tension: float
    mutual_inspiration: float
    idea_building_rate: float
    conflict_productivity: float
    
    def is_highly_synergistic(self) -> bool:
        """Check if pair has high creative synergy"""
        return self.synergy_score > 0.75 and self.mutual_inspiration > 0.7


class CollaborativeCreativityEngine:
    """Engine for managing collaborative creativity between agents"""
    
    def __init__(self):
        # Session management
        self.active_sessions: Dict[str, CreativeSession] = {}
        self.session_history: List[CreativeSession] = []
        self.creative_projects: Dict[str, CreativeProject] = {}
        
        # Creative dynamics
        self.participant_profiles: Dict[str, Dict[str, Any]] = {}
        self.synergy_matrix: Dict[Tuple[str, str], CreativeSynergy] = {}
        self.idea_network: Dict[str, List[str]] = {}  # Idea connections
        
        # Creative parameters
        self.novelty_threshold = 0.6
        self.synergy_amplification = 1.5
        self.diversity_bonus = 0.3
        self.flow_momentum = 0.8
        
        # Role effectiveness
        self.role_compatibility = {
            (CreativeRole.IDEATOR, CreativeRole.BUILDER): 0.9,
            (CreativeRole.IDEATOR, CreativeRole.CRITIC): 0.7,
            (CreativeRole.BUILDER, CreativeRole.REFINER): 0.8,
            (CreativeRole.EXPLORER, CreativeRole.SYNTHESIZER): 0.85,
            (CreativeRole.CATALYST, CreativeRole.IDEATOR): 0.9
        }
    
    def initiate_creative_session(self, participants: List[str], 
                                mode: CreativeMode,
                                focus_area: str) -> CreativeSession:
        """Start a new collaborative creative session"""
        # Create session
        session = CreativeSession(
            session_id=f"session_{datetime.now().timestamp()}",
            participants=participants,
            mode=mode,
            focus_area=focus_area,
            start_time=datetime.now(),
            ideas_generated=[],
            energy_flow=[(datetime.now(), CreativeEnergy.DIVERGENT)],
            breakthrough_moments=[],
            session_dynamics={
                'energy_level': 0.7,
                'harmony': 0.6,
                'productivity': 0.0,
                'innovation_rate': 0.0
            }
        )
        
        # Initialize participant roles
        self._assign_creative_roles(session)
        
        # Set up creative environment
        self._prepare_creative_space(session)
        
        self.active_sessions[session.session_id] = session
        return session
    
    def contribute_idea(self, session_id: str, contributor: str,
                       idea_content: str, idea_type: str = "general") -> CreativeIdea:
        """Contribute a new idea to the session"""
        if session_id not in self.active_sessions:
            raise ValueError(f"Session {session_id} not found")
        
        session = self.active_sessions[session_id]
        
        # Create idea
        idea = CreativeIdea(
            idea_id=f"idea_{datetime.now().timestamp()}",
            content=idea_content,
            originator=contributor,
            contributors=[contributor],
            creative_type=idea_type,
            novelty_score=self._assess_novelty(idea_content, session),
            feasibility_score=self._assess_feasibility(idea_content),
            impact_potential=self._assess_impact(idea_content, session),
            development_stage="initial",
            connections=[],
            metadata={
                'timestamp': datetime.now(),
                'session_context': session.focus_area,
                'energy_state': session.energy_flow[-1][1].value if session.energy_flow else None
            }
        )
        
        # Check for connections to existing ideas
        idea.connections = self._find_idea_connections(idea, session.ideas_generated)
        
        # Add to session
        session.ideas_generated.append(idea)
        
        # Update session dynamics
        self._update_session_dynamics(session, "idea_contributed")
        
        # Check for breakthrough
        if idea.get_overall_score() > 0.85:
            self._record_breakthrough(session, idea)
        
        return idea
    
    def build_on_idea(self, session_id: str, idea_id: str,
                     builder: str, enhancement: str) -> CreativeIdea:
        """Build upon an existing idea"""
        if session_id not in self.active_sessions:
            raise ValueError(f"Session {session_id} not found")
        
        session = self.active_sessions[session_id]
        
        # Find original idea
        original_idea = next((idea for idea in session.ideas_generated 
                            if idea.idea_id == idea_id), None)
        if not original_idea:
            raise ValueError(f"Idea {idea_id} not found")
        
        # Create enhanced idea
        enhanced_content = f"{original_idea.content} + {enhancement}"
        enhanced_idea = CreativeIdea(
            idea_id=f"idea_{datetime.now().timestamp()}",
            content=enhanced_content,
            originator=original_idea.originator,
            contributors=original_idea.contributors + [builder],
            creative_type=f"enhanced_{original_idea.creative_type}",
            novelty_score=self._assess_novelty(enhanced_content, session),
            feasibility_score=self._assess_feasibility(enhanced_content),
            impact_potential=self._assess_impact(enhanced_content, session) * 1.1,  # Bonus for building
            development_stage="enhanced",
            connections=[original_idea.idea_id],
            metadata={
                'timestamp': datetime.now(),
                'base_idea': idea_id,
                'enhancement_type': 'build_upon'
            }
        )
        
        # Add to session
        session.ideas_generated.append(enhanced_idea)
        
        # Update synergy between collaborators
        self._update_synergy(original_idea.originator, builder, "collaborative_building")
        
        return enhanced_idea
    
    def synthesize_ideas(self, session_id: str, idea_ids: List[str],
                        synthesizer: str) -> CreativeIdea:
        """Synthesize multiple ideas into a new concept"""
        if session_id not in self.active_sessions:
            raise ValueError(f"Session {session_id} not found")
        
        session = self.active_sessions[session_id]
        
        # Gather ideas to synthesize
        ideas_to_synthesize = [idea for idea in session.ideas_generated 
                              if idea.idea_id in idea_ids]
        
        if len(ideas_to_synthesize) < 2:
            raise ValueError("Need at least 2 ideas to synthesize")
        
        # Create synthesis
        synthesis_content = self._generate_synthesis(ideas_to_synthesize)
        all_contributors = list(set(sum([idea.contributors for idea in ideas_to_synthesize], [])))
        
        synthesized_idea = CreativeIdea(
            idea_id=f"idea_{datetime.now().timestamp()}",
            content=synthesis_content,
            originator=synthesizer,
            contributors=all_contributors + [synthesizer],
            creative_type="synthesis",
            novelty_score=self._assess_novelty(synthesis_content, session) * 1.2,  # Synthesis bonus
            feasibility_score=np.mean([idea.feasibility_score for idea in ideas_to_synthesize]),
            impact_potential=max([idea.impact_potential for idea in ideas_to_synthesize]) * 1.15,
            development_stage="synthesized",
            connections=idea_ids,
            metadata={
                'timestamp': datetime.now(),
                'synthesis_count': len(idea_ids),
                'synthesis_type': 'creative_merge'
            }
        )
        
        # Add to session
        session.ideas_generated.append(synthesized_idea)
        
        # Record as potential breakthrough
        if synthesized_idea.get_overall_score() > 0.8:
            self._record_breakthrough(session, synthesized_idea)
        
        return synthesized_idea
    
    def enter_creative_flow(self, session_id: str) -> CreativeFlow:
        """Attempt to enter collective creative flow state"""
        if session_id not in self.active_sessions:
            raise ValueError(f"Session {session_id} not found")
        
        session = self.active_sessions[session_id]
        
        # Assess current conditions
        flow_conditions = self._assess_flow_conditions(session)
        
        # Create flow state
        flow = CreativeFlow(
            flow_id=f"flow_{datetime.now().timestamp()}",
            participants=session.participants,
            flow_intensity=flow_conditions['intensity'],
            synchronization_level=flow_conditions['synchronization'],
            idea_velocity=self._calculate_idea_velocity(session),
            quality_consistency=self._assess_quality_consistency(session),
            disruption_tolerance=0.7,
            collective_focus=flow_conditions['focus']
        )
        
        # If conditions are right, enhance session
        if flow.is_in_flow():
            self._enhance_session_for_flow(session)
            session.energy_flow.append((datetime.now(), CreativeEnergy.TRANSFORMATIVE))
        
        return flow
    
    def facilitate_improvisation(self, session_id: str,
                               starting_point: str) -> List[CreativeIdea]:
        """Facilitate creative improvisation session"""
        if session_id not in self.active_sessions:
            raise ValueError(f"Session {session_id} not found")
        
        session = self.active_sessions[session_id]
        
        # Switch to improvisation mode
        session.mode = CreativeMode.IMPROVISATION
        session.energy_flow.append((datetime.now(), CreativeEnergy.DIVERGENT))
        
        # Generate improvisation chain
        improvisation_chain = []
        current_prompt = starting_point
        
        for participant in session.participants * 2:  # Two rounds
            # Each participant builds on previous
            improv_idea = self._generate_improvisation(
                participant, 
                current_prompt,
                session
            )
            
            idea = CreativeIdea(
                idea_id=f"improv_{datetime.now().timestamp()}",
                content=improv_idea,
                originator=participant,
                contributors=[participant],
                creative_type="improvisation",
                novelty_score=self._assess_novelty(improv_idea, session),
                feasibility_score=0.5,  # Improvisation focuses on creativity over feasibility
                impact_potential=self._assess_impact(improv_idea, session),
                development_stage="improvisational",
                connections=[improvisation_chain[-1].idea_id] if improvisation_chain else [],
                metadata={
                    'timestamp': datetime.now(),
                    'improvisation_round': len(improvisation_chain) + 1
                }
            )
            
            improvisation_chain.append(idea)
            session.ideas_generated.append(idea)
            current_prompt = improv_idea
        
        return improvisation_chain
    
    def create_project(self, project_name: str, participants: List[str],
                      project_type: str, objectives: List[str]) -> CreativeProject:
        """Create a new collaborative creative project"""
        project = CreativeProject(
            project_id=f"project_{datetime.now().timestamp()}",
            project_name=project_name,
            participants=participants,
            project_type=project_type,
            objectives=objectives,
            current_phase="ideation",
            ideas_pool=[],
            implementation_plan=[],
            milestones=self._generate_project_milestones(project_type, objectives),
            success_metrics={
                'innovation_level': 0.0,
                'collaboration_quality': 0.0,
                'objective_achievement': 0.0,
                'participant_satisfaction': 0.0
            }
        )
        
        self.creative_projects[project.project_id] = project
        return project
    
    def develop_project_idea(self, project_id: str, idea: CreativeIdea,
                           development_plan: Dict[str, Any]) -> Dict[str, Any]:
        """Develop an idea within a project context"""
        if project_id not in self.creative_projects:
            raise ValueError(f"Project {project_id} not found")
        
        project = self.creative_projects[project_id]
        
        # Add idea to project pool if not already there
        if idea not in project.ideas_pool:
            project.ideas_pool.append(idea)
        
        # Create development stages
        development_stages = self._create_development_stages(idea, development_plan)
        
        # Update idea stage
        idea.development_stage = "in_development"
        
        # Create implementation item
        implementation_item = {
            'idea_id': idea.idea_id,
            'development_stages': development_stages,
            'assigned_participants': development_plan.get('participants', project.participants),
            'timeline': development_plan.get('timeline', '4 weeks'),
            'resources_needed': development_plan.get('resources', []),
            'success_criteria': development_plan.get('success_criteria', [])
        }
        
        project.implementation_plan.append(implementation_item)
        
        return {
            'development_plan': implementation_item,
            'estimated_impact': idea.impact_potential,
            'next_steps': development_stages[0] if development_stages else None
        }
    
    def assess_creative_synergy(self, participant1: str, 
                              participant2: str) -> CreativeSynergy:
        """Assess creative synergy between two participants"""
        pair = tuple(sorted([participant1, participant2]))
        
        # Check if already assessed
        if pair in self.synergy_matrix:
            return self.synergy_matrix[pair]
        
        # Calculate synergy components
        complementarity = self._assess_complementarity(participant1, participant2)
        creative_tension = self._assess_creative_tension(participant1, participant2)
        mutual_inspiration = self._assess_mutual_inspiration(participant1, participant2)
        idea_building_rate = self._calculate_idea_building_rate(participant1, participant2)
        conflict_productivity = self._assess_conflict_productivity(participant1, participant2)
        
        # Overall synergy score
        synergy_score = (
            complementarity.get('overall', 0.5) * 0.3 +
            mutual_inspiration * 0.3 +
            idea_building_rate * 0.2 +
            creative_tension * 0.1 +
            conflict_productivity * 0.1
        )
        
        synergy = CreativeSynergy(
            participant_pair=pair,
            synergy_score=synergy_score,
            complementarity=complementarity,
            creative_tension=creative_tension,
            mutual_inspiration=mutual_inspiration,
            idea_building_rate=idea_building_rate,
            conflict_productivity=conflict_productivity
        )
        
        self.synergy_matrix[pair] = synergy
        return synergy
    
    def optimize_creative_team(self, available_participants: List[str],
                             project_requirements: Dict[str, Any],
                             team_size: int) -> List[str]:
        """Optimize team composition for creative project"""
        # Assess all possible combinations
        from itertools import combinations
        
        best_team = []
        best_score = 0.0
        
        for team in combinations(available_participants, team_size):
            team_score = self._evaluate_team_composition(
                list(team),
                project_requirements
            )
            
            if team_score > best_score:
                best_score = team_score
                best_team = list(team)
        
        return best_team
    
    def generate_creative_insights(self, session_id: str) -> Dict[str, Any]:
        """Generate insights from creative session"""
        if session_id not in self.active_sessions:
            if session_id in [s.session_id for s in self.session_history]:
                session = next(s for s in self.session_history if s.session_id == session_id)
            else:
                raise ValueError(f"Session {session_id} not found")
        else:
            session = self.active_sessions[session_id]
        
        insights = {
            'productivity_score': session.get_productivity_score(),
            'breakthrough_count': len(session.breakthrough_moments),
            'idea_quality_distribution': self._analyze_idea_quality(session),
            'collaboration_patterns': self._identify_collaboration_patterns(session),
            'energy_flow_analysis': self._analyze_energy_flow(session),
            'key_innovations': self._identify_key_innovations(session),
            'improvement_suggestions': self._generate_improvement_suggestions(session)
        }
        
        return insights
    
    def close_session(self, session_id: str) -> Dict[str, Any]:
        """Close a creative session and generate summary"""
        if session_id not in self.active_sessions:
            raise ValueError(f"Session {session_id} not found")
        
        session = self.active_sessions[session_id]
        
        # Generate session summary
        summary = {
            'session_id': session_id,
            'duration': datetime.now() - session.start_time,
            'participants': session.participants,
            'ideas_generated': len(session.ideas_generated),
            'top_ideas': sorted(session.ideas_generated, 
                              key=lambda i: i.get_overall_score(), 
                              reverse=True)[:5],
            'breakthrough_moments': session.breakthrough_moments,
            'productivity_score': session.get_productivity_score(),
            'collaboration_quality': self._assess_collaboration_quality(session),
            'next_steps': self._recommend_next_steps(session)
        }
        
        # Move to history
        self.session_history.append(session)
        del self.active_sessions[session_id]
        
        # Update participant profiles
        self._update_participant_profiles(session)
        
        return summary
    
    # Helper methods
    def _assign_creative_roles(self, session: CreativeSession) -> None:
        """Assign creative roles to participants"""
        available_roles = list(CreativeRole)
        
        for i, participant in enumerate(session.participants):
            # Assign based on participant profile or rotate
            if participant in self.participant_profiles:
                preferred_role = self.participant_profiles[participant].get('preferred_role')
                if preferred_role:
                    role = CreativeRole(preferred_role)
                else:
                    role = available_roles[i % len(available_roles)]
            else:
                role = available_roles[i % len(available_roles)]
            
            # Initialize profile if needed
            if participant not in self.participant_profiles:
                self.participant_profiles[participant] = {}
            
            self.participant_profiles[participant]['current_role'] = role
    
    def _prepare_creative_space(self, session: CreativeSession) -> None:
        """Prepare the creative environment for the session"""
        # Set initial energy based on mode
        if session.mode == CreativeMode.BRAINSTORMING:
            session.energy_flow.append((datetime.now(), CreativeEnergy.DIVERGENT))
        elif session.mode == CreativeMode.REFINEMENT:
            session.energy_flow.append((datetime.now(), CreativeEnergy.CONVERGENT))
        elif session.mode == CreativeMode.INNOVATION:
            session.energy_flow.append((datetime.now(), CreativeEnergy.DISRUPTIVE))
    
    def _assess_novelty(self, idea_content: str, session: CreativeSession) -> float:
        """Assess the novelty of an idea"""
        # Simple implementation - could use more sophisticated NLP
        base_novelty = 0.5
        
        # Check against existing ideas
        similarity_scores = []
        for existing_idea in session.ideas_generated:
            similarity = self._calculate_similarity(idea_content, existing_idea.content)
            similarity_scores.append(similarity)
        
        if similarity_scores:
            max_similarity = max(similarity_scores)
            novelty = 1.0 - max_similarity
        else:
            novelty = 0.7  # First idea gets good novelty
        
        # Adjust for session context
        if session.mode == CreativeMode.INNOVATION:
            novelty *= 1.2  # Boost novelty in innovation mode
        
        return min(novelty, 1.0)
    
    def _assess_feasibility(self, idea_content: str) -> float:
        """Assess the feasibility of an idea"""
        # Simple heuristic - could be enhanced
        feasibility = 0.6
        
        # Penalize overly complex ideas
        if len(idea_content) > 200:
            feasibility *= 0.9
        
        # Check for implementation keywords
        implementation_keywords = ['simple', 'straightforward', 'existing', 'proven']
        if any(keyword in idea_content.lower() for keyword in implementation_keywords):
            feasibility *= 1.1
        
        return min(feasibility, 1.0)
    
    def _assess_impact(self, idea_content: str, session: CreativeSession) -> float:
        """Assess the potential impact of an idea"""
        base_impact = 0.5
        
        # Check for impact keywords
        impact_keywords = ['transform', 'revolutionize', 'significant', 'breakthrough', 'novel']
        keyword_count = sum(1 for keyword in impact_keywords if keyword in idea_content.lower())
        
        impact = base_impact + (keyword_count * 0.1)
        
        # Adjust for session focus
        if session.focus_area in idea_content:
            impact *= 1.2
        
        return min(impact, 1.0)
    
    def _find_idea_connections(self, new_idea: CreativeIdea, 
                              existing_ideas: List[CreativeIdea]) -> List[str]:
        """Find connections between ideas"""
        connections = []
        
        for existing_idea in existing_ideas:
            similarity = self._calculate_similarity(new_idea.content, existing_idea.content)
            if similarity > 0.3 and similarity < 0.8:  # Related but not duplicate
                connections.append(existing_idea.idea_id)
        
        return connections
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts"""
        # Simple word overlap - could use embeddings
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)
    
    def _update_session_dynamics(self, session: CreativeSession, event: str) -> None:
        """Update session dynamics based on events"""
        if event == "idea_contributed":
            session.session_dynamics['productivity'] = len(session.ideas_generated) / 10
            session.session_dynamics['innovation_rate'] = self._calculate_innovation_rate(session)
    
    def _calculate_innovation_rate(self, session: CreativeSession) -> float:
        """Calculate rate of innovation in session"""
        if not session.ideas_generated:
            return 0.0
        
        high_novelty_ideas = [idea for idea in session.ideas_generated 
                             if idea.novelty_score > self.novelty_threshold]
        
        return len(high_novelty_ideas) / len(session.ideas_generated)
    
    def _record_breakthrough(self, session: CreativeSession, idea: CreativeIdea) -> None:
        """Record a breakthrough moment"""
        breakthrough = {
            'timestamp': datetime.now(),
            'idea_id': idea.idea_id,
            'breakthrough_type': 'high_quality_idea',
            'participants_involved': idea.contributors,
            'impact_score': idea.get_overall_score()
        }
        
        session.breakthrough_moments.append(breakthrough)
        
        # Boost session energy
        session.energy_flow.append((datetime.now(), CreativeEnergy.TRANSFORMATIVE))
    
    def _update_synergy(self, participant1: str, participant2: str, 
                       interaction_type: str) -> None:
        """Update synergy between participants based on interaction"""
        pair = tuple(sorted([participant1, participant2]))
        
        if pair not in self.synergy_matrix:
            # Create initial synergy
            self.assess_creative_synergy(participant1, participant2)
        
        synergy = self.synergy_matrix[pair]
        
        # Update based on interaction
        if interaction_type == "collaborative_building":
            synergy.idea_building_rate = min(synergy.idea_building_rate * 1.1, 1.0)
            synergy.mutual_inspiration = min(synergy.mutual_inspiration * 1.05, 1.0)
    
    def _generate_synthesis(self, ideas: List[CreativeIdea]) -> str:
        """Generate synthesis from multiple ideas"""
        # Extract key concepts from each idea
        key_concepts = []
        for idea in ideas:
            # Simple extraction - could use NLP
            words = idea.content.split()
            key_concepts.extend(words[:5])  # Take first 5 words as key concepts
        
        # Create synthesis
        synthesis = f"Synthesis combining: {', '.join(set(key_concepts))}"
        return synthesis
    
    def _assess_flow_conditions(self, session: CreativeSession) -> Dict[str, float]:
        """Assess conditions for creative flow"""
        conditions = {
            'intensity': session.session_dynamics.get('energy_level', 0.5),
            'synchronization': self._calculate_synchronization(session),
            'focus': self._calculate_collective_focus(session)
        }
        
        return conditions
    
    def _calculate_synchronization(self, session: CreativeSession) -> float:
        """Calculate synchronization level among participants"""
        if len(session.participants) < 2:
            return 1.0
        
        # Check idea contribution patterns
        contribution_times = []
        for idea in session.ideas_generated[-10:]:  # Last 10 ideas
            if 'timestamp' in idea.metadata:
                contribution_times.append(idea.metadata['timestamp'])
        
        if len(contribution_times) < 2:
            return 0.5
        
        # Calculate time intervals
        intervals = []
        for i in range(1, len(contribution_times)):
            interval = (contribution_times[i] - contribution_times[i-1]).total_seconds()
            intervals.append(interval)
        
        # Low variance in intervals indicates synchronization
        if intervals:
            variance = np.var(intervals)
            synchronization = 1.0 / (1.0 + variance / 100)  # Normalize
            return synchronization
        
        return
