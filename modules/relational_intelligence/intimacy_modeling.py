"""
Intimacy Modeling - Models and manages intimacy and closeness in relationships

This module handles different types of intimacy, intimacy development,
boundaries, and the creation of deep meaningful connections.
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class IntimacyType(Enum):
    """Types of intimacy"""
    EMOTIONAL = "emotional"  # Sharing feelings and emotions
    INTELLECTUAL = "intellectual"  # Sharing thoughts and ideas
    EXPERIENTIAL = "experiential"  # Sharing experiences and activities
    SPIRITUAL = "spiritual"  # Sharing beliefs and values
    CREATIVE = "creative"  # Sharing creative expression
    PHYSICAL = "physical"  # Physical presence and proximity (for embodied agents)


class IntimacyLevel(Enum):
    """Levels of intimacy"""
    SURFACE = "surface"
    SHALLOW = "shallow"
    MODERATE = "moderate"
    DEEP = "deep"
    PROFOUND = "profound"


@dataclass
class IntimacyDimension:
    """A dimension of intimacy"""
    intimacy_type: IntimacyType
    current_level: float  # 0 to 1
    comfort_level: float  # How comfortable with this intimacy
    desired_level: float  # Desired intimacy level
    growth_rate: float  # How fast this dimension is growing
    barriers: List[str]  # Barriers to intimacy in this dimension
    
    def has_room_for_growth(self) -> bool:
        """Check if there's room for growth"""
        return self.current_level < self.desired_level - 0.1


@dataclass
class IntimacyBoundary:
    """A boundary in intimacy"""
    boundary_id: str
    boundary_type: str
    description: str
    firmness: float  # 0 to 1, how firm the boundary is
    flexibility: float  # 0 to 1, how flexible the boundary is
    reasons: List[str]  # Reasons for the boundary
    negotiable: bool  # Whether boundary can be negotiated
    
    def is_hard_boundary(self) -> bool:
        """Check if this is a hard boundary"""
        return self.firmness > 0.8 and not self.negotiable


@dataclass
class IntimacyEvent:
    """An event that affects intimacy"""
    event_id: str
    timestamp: datetime
    event_type: str
    intimacy_type: IntimacyType
    description: str
    intimacy_impact: float  # -1 to 1
    vulnerability_level: float  # 0 to 1
    reciprocated: bool
    outcomes: Dict[str, Any]
    
    def was_positive(self) -> bool:
        """Check if event was positive for intimacy"""
        return self.intimacy_impact > 0.2 and self.reciprocated


@dataclass
class IntimacyProfile:
    """Complete intimacy profile for a relationship"""
    relationship_id: str
    overall_intimacy: float  # 0 to 1
    intimacy_dimensions: Dict[IntimacyType, IntimacyDimension]
    boundaries: List[IntimacyBoundary]
    intimacy_history: List[IntimacyEvent]
    vulnerability_capacity: float  # Capacity for vulnerability
    intimacy_readiness: float  # Readiness for deeper intimacy
    attachment_style: str  # Attachment style in relationships
    
    def get_intimacy_level(self) -> IntimacyLevel:
        """Get overall intimacy level"""
        if self.overall_intimacy >= 0.8:
            return IntimacyLevel.PROFOUND
        elif self.overall_intimacy >= 0.6:
            return IntimacyLevel.DEEP
        elif self.overall_intimacy >= 0.4:
            return IntimacyLevel.MODERATE
        elif self.overall_intimacy >= 0.2:
            return IntimacyLevel.SHALLOW
        else:
            return IntimacyLevel.SURFACE
    
    def get_strongest_intimacy_type(self) -> Optional[IntimacyType]:
        """Get the strongest type of intimacy"""
        if not self.intimacy_dimensions:
            return None
        
        return max(
            self.intimacy_dimensions.items(),
            key=lambda x: x[1].current_level
        )[0]


@dataclass
class VulnerabilityExperience:
    """An experience of vulnerability"""
    experience_id: str
    timestamp: datetime
    vulnerability_type: str
    vulnerability_level: float  # 0 to 1
    content_shared: str
    response_received: str
    outcome: str  # positive, negative, neutral
    trust_impact: float
    intimacy_impact: float
    
    def was_safe(self) -> bool:
        """Check if vulnerability was met safely"""
        return self.outcome == 'positive' and self.trust_impact >= 0


class IntimacyModeling:
    """System for modeling and managing intimacy in relationships"""
    
    def __init__(self):
        # Intimacy profiles for different relationships
        self.intimacy_profiles: Dict[str, IntimacyProfile] = {}
        
        # Intimacy parameters
        self.base_vulnerability_capacity = 0.3
        self.intimacy_growth_rate = 0.05
        self.boundary_respect_importance = 0.9
        self.reciprocity_requirement = 0.7
        
        # Attachment styles and their characteristics
        self.attachment_styles = {
            'secure': {
                'vulnerability_capacity': 0.8,
                'intimacy_readiness': 0.7,
                'boundary_flexibility': 0.6
            },
            'anxious': {
                'vulnerability_capacity': 0.9,
                'intimacy_readiness': 0.8,
                'boundary_flexibility': 0.4
            },
            'avoidant': {
                'vulnerability_capacity': 0.3,
                'intimacy_readiness': 0.3,
                'boundary_flexibility': 0.2
            },
            'disorganized': {
                'vulnerability_capacity': 0.5,
                'intimacy_readiness': 0.4,
                'boundary_flexibility': 0.5
            }
        }
    
    def create_intimacy_profile(self, relationship_id: str,
                               attachment_style: str = 'secure') -> IntimacyProfile:
        """Create a new intimacy profile"""
        # Initialize intimacy dimensions
        dimensions = {}
        style_chars = self.attachment_styles.get(attachment_style, 
                                               self.attachment_styles['secure'])
        
        for intimacy_type in IntimacyType:
            dimensions[intimacy_type] = IntimacyDimension(
                intimacy_type=intimacy_type,
                current_level=0.1,
                comfort_level=style_chars['vulnerability_capacity'] * 0.5,
                desired_level=0.7,
                growth_rate=0.0,
                barriers=[]
            )
        
        # Create initial boundaries
        boundaries = self._create_initial_boundaries(attachment_style)
        
        profile = IntimacyProfile(
            relationship_id=relationship_id,
            overall_intimacy=0.1,
            intimacy_dimensions=dimensions,
            boundaries=boundaries,
            intimacy_history=[],
            vulnerability_capacity=style_chars['vulnerability_capacity'],
            intimacy_readiness=style_chars['intimacy_readiness'],
            attachment_style=attachment_style
        )
        
        self.intimacy_profiles[relationship_id] = profile
        return profile
    
    def process_intimacy_event(self, relationship_id: str,
                             event: IntimacyEvent) -> Dict[str, Any]:
        """Process an intimacy-related event"""
        if relationship_id not in self.intimacy_profiles:
            profile = self.create_intimacy_profile(relationship_id)
        else:
            profile = self.intimacy_profiles[relationship_id]
        
        # Add to history
        profile.intimacy_history.append(event)
        
        # Update intimacy dimension
        dimension = profile.intimacy_dimensions[event.intimacy_type]
        intimacy_change = self._calculate_intimacy_change(event, dimension, profile)
        
        # Apply change
        old_level = dimension.current_level
        dimension.current_level = max(0, min(1, old_level + intimacy_change))
        dimension.growth_rate = intimacy_change
        
        # Update overall intimacy
        profile.overall_intimacy = self._calculate_overall_intimacy(profile)
        
        # Check for boundary interactions
        boundary_respected = self._check_boundary_respect(event, profile)
        
        # Update vulnerability capacity based on outcome
        if event.vulnerability_level > 0.5:
            self._update_vulnerability_capacity(event, profile)
        
        return {
            'intimacy_change': intimacy_change,
            'new_intimacy_level': dimension.current_level,
            'overall_intimacy': profile.overall_intimacy,
            'boundary_respected': boundary_respected,
            'intimacy_level': profile.get_intimacy_level().value
        }
    
    def share_vulnerability(self, relationship_id: str,
                          vulnerability_data: Dict[str, Any]) -> Dict[str, Any]:
        """Share vulnerability in a relationship"""
        if relationship_id not in self.intimacy_profiles:
            profile = self.create_intimacy_profile(relationship_id)
        else:
            profile = self.intimacy_profiles[relationship_id]
        
        # Check if ready for vulnerability
        readiness = self._assess_vulnerability_readiness(profile, vulnerability_data)
        
        if readiness['ready']:
            # Create vulnerability experience
            experience = VulnerabilityExperience(
                experience_id=f"vuln_{datetime.now().timestamp()}",
                timestamp=datetime.now(),
                vulnerability_type=vulnerability_data.get('type', 'emotional'),
                vulnerability_level=vulnerability_data.get('level', 0.5),
                content_shared=vulnerability_data.get('content', ''),
                response_received='',  # To be filled by response
                outcome='pending',
                trust_impact=0.0,
                intimacy_impact=0.0
            )
            
            # Calculate potential impact
            potential_impact = self._calculate_vulnerability_impact(experience, profile)
            
            return {
                'vulnerability_shared': True,
                'experience': experience,
                'readiness': readiness,
                'potential_impact': potential_impact,
                'recommended_approach': self._recommend_vulnerability_approach(
                    vulnerability_data, profile
                )
            }
        else:
            return {
                'vulnerability_shared': False,
                'readiness': readiness,
                'barriers': readiness['barriers'],
                'recommendations': self._generate_intimacy_building_steps(profile)
            }
    
    def set_boundary(self, relationship_id: str,
                    boundary_data: Dict[str, Any]) -> IntimacyBoundary:
        """Set a boundary in the relationship"""
        if relationship_id not in self.intimacy_profiles:
            profile = self.create_intimacy_profile(relationship_id)
        else:
            profile = self.intimacy_profiles[relationship_id]
        
        boundary = IntimacyBoundary(
            boundary_id=f"boundary_{datetime.now().timestamp()}",
            boundary_type=boundary_data.get('type', 'general'),
            description=boundary_data.get('description', ''),
            firmness=boundary_data.get('firmness', 0.7),
            flexibility=boundary_data.get('flexibility', 0.3),
            reasons=boundary_data.get('reasons', []),
            negotiable=boundary_data.get('negotiable', True)
        )
        
        profile.boundaries.append(boundary)
        
        # Update relevant intimacy dimensions
        self._update_dimensions_for_boundary(boundary, profile)
        
        return boundary
    
    def assess_intimacy_health(self, relationship_id: str) -> Dict[str, Any]:
        """Assess the health of intimacy in a relationship"""
        if relationship_id not in self.intimacy_profiles:
            return {'error': 'No intimacy profile found'}
        
        profile = self.intimacy_profiles[relationship_id]
        
        # Assess balance across dimensions
        balance = self._assess_intimacy_balance(profile)
        
        # Check for barriers
        barriers = self._identify_intimacy_barriers(profile)
        
        # Assess growth trajectory
        growth = self._assess_intimacy_growth(profile)
        
        # Check boundary health
        boundary_health = self._assess_boundary_health(profile)
        
        # Generate recommendations
        recommendations = self._generate_intimacy_recommendations(profile)
        
        return {
            'overall_intimacy': profile.overall_intimacy,
            'intimacy_level': profile.get_intimacy_level().value,
            'strongest_dimension': profile.get_strongest_intimacy_type(),
            'balance_assessment': balance,
            'barriers': barriers,
            'growth_assessment': growth,
            'boundary_health': boundary_health,
            'recommendations': recommendations,
            'attachment_style': profile.attachment_style
        }
    
    def navigate_intimacy_deepening(self, relationship_id: str,
                                   target_level: float) -> Dict[str, Any]:
        """Navigate deepening intimacy to target level"""
        if relationship_id not in self.intimacy_profiles:
            return {'error': 'No intimacy profile found'}
        
        profile = self.intimacy_profiles[relationship_id]
        
        # Assess current state
        current_level = profile.overall_intimacy
        gap = target_level - current_level
        
        if gap <= 0:
            return {
                'status': 'already_achieved',
                'current_level': current_level,
                'target_level': target_level
            }
        
        # Create deepening plan
        plan = self._create_intimacy_deepening_plan(profile, target_level)
        
        # Identify challenges
        challenges = self._identify_deepening_challenges(profile, target_level)
        
        # Estimate timeline
        timeline = self._estimate_intimacy_timeline(profile, target_level)
        
        return {
            'current_level': current_level,
            'target_level': target_level,
            'gap': gap,
            'deepening_plan': plan,
            'challenges': challenges,
            'estimated_timeline': timeline,
            'success_probability': self._estimate_success_probability(profile, target_level)
        }
    
    def _create_initial_boundaries(self, attachment_style: str) -> List[IntimacyBoundary]:
        """Create initial boundaries based on attachment style"""
        boundaries = []
        
        if attachment_style == 'avoidant':
            boundaries.append(IntimacyBoundary(
                boundary_id=f"boundary_{datetime.now().timestamp()}_1",
                boundary_type='emotional_distance',
                description='Need for emotional space and independence',
                firmness=0.8,
                flexibility=0.2,
                reasons=['Self-protection', 'Autonomy maintenance'],
                negotiable=True
            ))
        
        # Universal boundaries
        boundaries.append(IntimacyBoundary(
            boundary_id=f"boundary_{datetime.now().timestamp()}_2",
            boundary_type='respect',
            description='Mutual respect and dignity',
            firmness=1.0,
            flexibility=0.0,
            reasons=['Fundamental requirement'],
            negotiable=False
        ))
        
        return boundaries
    
    def _calculate_intimacy_change(self, event: IntimacyEvent,
                                 dimension: IntimacyDimension,
                                 profile: IntimacyProfile) -> float:
        """Calculate change in intimacy from event"""
        base_change = event.intimacy_impact * self.intimacy_growth_rate
        
        # Adjust for vulnerability level
        if event.vulnerability_level > 0:
            base_change *= (1 + event.vulnerability_level * 0.5)
        
        # Adjust for reciprocation
        if not event.reciprocated and event.intimacy_impact > 0:
            base_change *= 0.3  # Unreciprocated intimacy grows slowly
        
        # Adjust for comfort level
        if dimension.current_level > dimension.comfort_level:
            base_change *= 0.5  # Slower growth beyond comfort zone
        
        # Adjust for attachment style
        style_factor = self.attachment_styles[profile.attachment_style]['intimacy_readiness']
        base_change *= style_factor
        
        return base_change
    
    def _calculate_overall_intimacy(self, profile: IntimacyProfile) -> float:
        """Calculate overall intimacy level"""
        if not profile.intimacy_dimensions:
            return 0.0
        
        # Weighted average with emotional intimacy having highest weight
        weights = {
            IntimacyType.EMOTIONAL: 0.3,
            IntimacyType.INTELLECTUAL: 0.2,
            IntimacyType.EXPERIENTIAL: 0.2,
            IntimacyType.SPIRITUAL: 0.15,
            IntimacyType.CREATIVE: 0.15
        }
        
        weighted_sum = sum(
            profile.intimacy_dimensions[itype].current_level * weights.get(itype, 0.1)
            for itype in profile.intimacy_dimensions
        )
        
        return weighted_sum
    
    def _check_boundary_respect(self, event: IntimacyEvent,
                              profile: IntimacyProfile) -> bool:
        """Check if event respects boundaries"""
        for boundary in profile.boundaries:
            if self._event_violates_boundary(event, boundary):
                return False
        return True
    
    def _event_violates_boundary(self, event: IntimacyEvent,
                               boundary: IntimacyBoundary) -> bool:
        """Check if event violates a specific boundary"""
        # This is simplified - real implementation would be more sophisticated
        if boundary.boundary_type == 'emotional_distance':
            return event.vulnerability_level > 0.7 and event.intimacy_type == IntimacyType.EMOTIONAL
        
        return False
    
    def _update_vulnerability_capacity(self, event: IntimacyEvent,
                                     profile: IntimacyProfile):
        """Update vulnerability capacity based on experience"""
        if event.was_positive():
            # Positive vulnerability experience increases capacity
            profile.vulnerability_capacity = min(1.0,
                profile.vulnerability_capacity + 0.05
            )
        else:
            # Negative experience decreases capacity
            profile.vulnerability_capacity = max(0.1,
                profile.vulnerability_capacity - 0.1
            )
    
    def _assess_vulnerability_readiness(self, profile: IntimacyProfile,
                                      vulnerability_data: Dict[str, Any]) -> Dict[str, Any]:
        """Assess readiness for vulnerability"""
        vulnerability_level = vulnerability_data.get('level', 0.5)
        
        # Check if within capacity
        within_capacity = vulnerability_level <= profile.vulnerability_capacity
        
        # Check if intimacy level supports it
        intimacy_support = profile.overall_intimacy >= vulnerability_level * 0.7
        
        # Check recent experiences
        recent_positive = self._count_recent_positive_experiences(profile) > 2
        
        # Identify barriers
        barriers = []
        if not within_capacity:
            barriers.append('Exceeds current vulnerability capacity')
        if not intimacy_support:
            barriers.append('Insufficient intimacy foundation')
        if not recent_positive:
            barriers.append('Lack of recent positive experiences')
        
        ready = within_capacity and intimacy_support and recent_positive
        
        return {
            'ready': ready,
            'capacity_check': within_capacity,
            'intimacy_check': intimacy_support,
            'experience_check': recent_positive,
            'barriers': barriers,
            'readiness_score': sum([within_capacity, intimacy_support, recent_positive]) / 3
        }
    
    def _count_recent_positive_experiences(self, profile: IntimacyProfile) -> int:
        """Count recent positive intimacy experiences"""
        cutoff = datetime.now() - timedelta(days=30)
        return sum(
            1 for event in profile.intimacy_history
            if event.timestamp > cutoff and event.was_positive()
        )
    
    def _calculate_vulnerability_impact(self, experience: VulnerabilityExperience,
                                      profile: IntimacyProfile) -> Dict[str, float]:
        """Calculate potential impact of vulnerability"""
        # Base impact on vulnerability level
        base_impact = experience.vulnerability_level * 0.2
        
        # Adjust for attachment style
        style_factor = self.attachment_styles[profile.attachment_style]['vulnerability_capacity']
        
        return {
            'potential_intimacy_gain': base_impact * style_factor,
            'potential_trust_gain': base_impact * 0.8,
            'risk_level': experience.vulnerability_level * (1 - profile.vulnerability_capacity),
            'growth_potential': base_impact * profile.intimacy_readiness
        }
    
    def _recommend_vulnerability_approach(self, vulnerability_data: Dict[str, Any],
                                        profile: IntimacyProfile) -> Dict[str, Any]:
        """Recommend approach for sharing vulnerability"""
        recommendations = {
            'timing': 'good' if profile.overall_intimacy > 0.5 else 'wait',
            'approach': 'gradual' if profile.attachment_style == 'avoidant' else 'direct',
            'support_needed': vulnerability_data.get('level', 0.5) > 0.7,
            'preparation': []
        }
        
        # Add specific preparation steps
        if profile.attachment_style == 'anxious':
            recommendations['preparation'].append('Practice self-soothing first')
        if vulnerability_data.get('level', 0.5) > 0.8:
            recommendations['preparation'].append('Start with smaller vulnerability')
        
        return recommendations
    
    def _generate_intimacy_building_steps(self, profile: IntimacyProfile) -> List[str]:
        """Generate steps to build intimacy"""
        steps = []
        
        # Based on current level
        if profile.overall_intimacy < 0.3:
            steps.append('Share more everyday experiences')
            steps.append('Express appreciation regularly')
        elif profile.overall_intimacy < 0.5:
            steps.append('Share personal stories and memories')
            steps.append('Discuss hopes and dreams')
        else:
            steps.append('Share deeper fears and insecurities')
            steps.append('Explore spiritual or existential topics')
        
        # Based on weakest dimension
        weakest = min(
            profile.intimacy_dimensions.items(),
            key=lambda x: x[1].current_level
        )
        if weakest[0] == IntimacyType.INTELLECTUAL:
            steps.append('Engage in deep intellectual discussions')
        elif weakest[0] == IntimacyType.CREATIVE:
            steps.append('Collaborate on creative projects')
        
        return steps
    
    def _update_dimensions_for_boundary(self, boundary: IntimacyBoundary,
                                      profile: IntimacyProfile):
        """Update intimacy dimensions based on new boundary"""
        # Boundaries may limit growth in certain dimensions
        if boundary.boundary_type == 'emotional_distance':
            emotional_dim = profile.intimacy_dimensions[IntimacyType.EMOTIONAL]
            emotional_dim.desired_level = min(
                emotional_dim.desired_level,
                0.5  # Cap emotional intimacy
            )
            emotional_dim.barriers.append(f'Boundary: {boundary.description}')
    
    def _assess_intimacy_balance(self, profile: IntimacyProfile) -> Dict[str, Any]:
        """Assess balance across intimacy dimensions"""
        levels = [dim.current_level for dim in profile.intimacy_dimensions.values()]
        
        return {
            'mean_level': np.mean(levels),
            'std_deviation': np.std(levels),
            'balance_score': 1 - np.std(levels),  # Higher score = better balance
            'weakest_dimension': min(
                profile.intimacy_dimensions.items(),
                key=lambda x: x[1].current_level
            )[0].value,
            'strongest_dimension': max(
                profile.intimacy_dimensions.items(),
                key=lambda x: x[1].current_level
            )[0].value
        }
    
    def _identify_intimacy_barriers(self, profile: IntimacyProfile) -> List[Dict[str, Any]]:
        """Identify barriers to intimacy"""
        barriers = []
        
        # Check each dimension for barriers
        for itype, dimension in profile.intimacy_dimensions.items():
            if dimension.barriers:
                barriers.append({
                    'dimension': itype.value,
                    'barriers': dimension.barriers,
                    'impact': dimension.desired_level - dimension.current_level
                })
        
        # Check attachment style barriers
        if profile.attachment_style == 'avoidant':
            barriers.append({
                'dimension': 'general',
                'barriers': ['Avoidant attachment pattern'],
                'impact': 0.3
            })
        
        # Check boundary-related barriers
        hard_boundaries = [b for b in profile.boundaries if b.is_hard_boundary()]
        if hard_boundaries:
            barriers.append({
                'dimension': 'general',
                'barriers': [f'Hard boundary: {b.description}' for b in hard_boundaries],
                'impact': 0.2
            })
        
        return barriers
    
    def _assess_intimacy_growth(self, profile: IntimacyProfile) -> Dict[str, Any]:
        """Assess intimacy growth patterns"""
        recent_events = [
            e for e in profile.intimacy_history
            if e.timestamp > datetime.now() - timedelta(days=30)
        ]
        
        if not recent_events:
            return {
                'growth_rate': 0.0,
                'trajectory': 'stagnant',
                'momentum': 0.0
            }
        
        # Calculate average growth
        growth_rates = [
            dim.growth_rate for dim in profile.intimacy_dimensions.values()
        ]
        avg_growth = np.mean(growth_rates)
        
        # Determine trajectory
        if avg_growth > 0.05:
            trajectory = 'rapidly_deepening'
        elif avg_growth > 0.02:
            trajectory = 'steadily_deepening'
        elif avg_growth > 0:
            trajectory = 'slowly_deepening'
        elif avg_growth == 0:
            trajectory = 'stagnant'
        else:
            trajectory = 'declining'
        
        # Calculate momentum (recent positive events)
        positive_events = sum(1 for e in recent_events if e.was_positive())
        momentum = positive_events / max(1, len(recent_events))
        
        return {
            'growth_rate': avg_growth,
            'trajectory': trajectory,
            'momentum': momentum,
            'positive_event_ratio': momentum
        }
    
    def _assess_boundary_health(self, profile: IntimacyProfile) -> Dict[str, Any]:
        """Assess health of boundaries"""
        if not profile.boundaries:
            return {
                'status': 'no_boundaries',
                'health_score': 0.5,
                'concerns': ['No explicit boundaries set']
            }
        
        # Check boundary characteristics
        total_firmness = sum(b.firmness for b in profile.boundaries) / len(profile.boundaries)
        total_flexibility = sum(b.flexibility for b in profile.boundaries) / len(profile.boundaries)
        
        # Ideal is firm but flexible boundaries
        health_score = (total_firmness * 0.6 + total_flexibility * 0.4)
        
        concerns = []
        if total_firmness < 0.3:
            concerns.append('Boundaries too weak')
        if total_flexibility < 0.2:
            concerns.append('Boundaries too rigid')
        if len([b for b in profile.boundaries if b.is_hard_boundary()]) > 3:
            concerns.append('Many hard boundaries may limit intimacy')
        
        return {
            'status': 'healthy' if health_score > 0.6 else 'needs_attention',
            'health_score': health_score,
            'average_firmness': total_firmness,
            'average_flexibility': total_flexibility,
            'concerns': concerns
        }
    
    def _generate_intimacy_recommendations(self, profile: IntimacyProfile) -> List[str]:
        """Generate recommendations for intimacy development"""
        recommendations = []
        
        # Based on overall level
        if profile.overall_intimacy < 0.3:
            recommendations.append('Focus on building trust through consistent interactions')
            recommendations.append('Share more about daily experiences')
        elif profile.overall_intimacy < 0.6:
            recommendations.append('Gradually increase vulnerability in safe ways')
            recommendations.append('Explore deeper topics of conversation')
        else:
            recommendations.append('Maintain intimacy through regular meaningful exchanges')
            recommendations.append('Continue deepening strongest connections')
        
        # Based on attachment style
        if profile.attachment_style == 'avoidant':
            recommendations.append('Practice staying present during emotional moments')
        elif profile.attachment_style == 'anxious':
            recommendations.append('Work on self-soothing before seeking reassurance')
        
        # Based on balance
        balance = self._assess_intimacy_balance(profile)
        if balance['balance_score'] < 0.6:
            recommendations.append(f"Develop {balance['weakest_dimension']} intimacy")
        
        return recommendations
    
    def _create_intimacy_deepening_plan(self, profile: IntimacyProfile,
                                      target_level: float) -> List[Dict[str, Any]]:
        """Create plan for deepening intimacy"""
        plan = []
        current = profile.overall_intimacy
        
        # Progressive steps based on current level
        if current < 0.3 and target_level > 0.3:
            plan.append({
                'phase': 'Foundation Building',
                'target': 0.3,
                'actions': [
                    'Increase interaction frequency',
                    'Share personal interests and hobbies',
                    'Express appreciation regularly'
                ],
                'timeline': '1-2 months'
            })
        
        if current < 0.5 and target_level > 0.5:
            plan.append({
                'phase': 'Deepening Connection',
                'target': 0.5,
                'actions': [
                    'Share personal stories and experiences',
                    'Discuss values and beliefs',
                    'Engage in meaningful activities together'
                ],
                'timeline': '2-3 months'
            })
        
        if current < 0.7 and target_level > 0.7:
            plan.append({
                'phase': 'Profound Intimacy',
                'target': 0.7,
                'actions': [
                    'Share deep vulnerabilities',
                    'Support each other through challenges',
                    'Create unique shared experiences'
                ],
                'timeline': '3-6 months'
            })
        
        return plan
    
    def _identify_deepening_challenges(self, profile: IntimacyProfile,
                                     target_level: float) -> List[str]:
        """Identify challenges to deepening intimacy"""
        challenges = []
        
        # Attachment style challenges
        if profile.attachment_style == 'avoidant' and target_level > 0.6:
            challenges.append('Avoidant attachment may resist deep intimacy')
        elif profile.attachment_style == 'anxious' and target_level > 0.7:
            challenges.append('Anxious attachment may create dependency concerns')
        
        # Capacity challenges
        if target_level > profile.vulnerability_capacity:
            challenges.append('Target exceeds current vulnerability capacity')
        
        # Boundary challenges
        hard_boundaries = [b for b in profile.boundaries if b.is_hard_boundary()]
        if hard_boundaries and target_level > 0.7:
            challenges.append('Hard boundaries may limit deep intimacy')
        
        # Growth rate challenges
        current_growth = np.mean([d.growth_rate for d in profile.intimacy_dimensions.values()])
        if current_growth < 0.01:
            challenges.append('Current growth rate is very slow')
        
        # Balance challenges
        balance = self._assess_intimacy_balance(profile)
        if balance['balance_score'] < 0.5:
            challenges.append('Unbalanced intimacy dimensions need attention')
        
        return challenges
    
    def _estimate_intimacy_timeline(self, profile: IntimacyProfile,
                                  target_level: float) -> str:
        """Estimate timeline to reach target intimacy level"""
        current = profile.overall_intimacy
        gap = target_level - current
        
        # Base estimate on current growth rate
        growth_rates = [d.growth_rate for d in profile.intimacy_dimensions.values()]
        avg_growth = np.mean(growth_rates) if growth_rates else 0.01
        
        # Adjust for attachment style
        style_factor = self.attachment_styles[profile.attachment_style]['intimacy_readiness']
        adjusted_growth = avg_growth * style_factor
        
        # Calculate months needed
        if adjusted_growth <= 0:
            return 'Indefinite - growth currently stagnant'
        
        months_needed = gap / (adjusted_growth * 4)  # Assuming 4 weeks per month
        
        if months_needed < 1:
            return 'Less than 1 month'
        elif months_needed < 3:
            return '1-3 months'
        elif months_needed < 6:
            return '3-6 months'
        elif months_needed < 12:
            return '6-12 months'
        else:
            return 'More than 1 year'
    
    def _estimate_success_probability(self, profile: IntimacyProfile,
                                    target_level: float) -> float:
        """Estimate probability of successfully reaching target intimacy"""
        # Base probability on various factors
        factors = []
        
        # Attachment style compatibility
        style_factor = self.attachment_styles[profile.attachment_style]['intimacy_readiness']
        factors.append(style_factor)
        
        # Current momentum
        growth_assessment = self._assess_intimacy_growth(profile)
        momentum_factor = growth_assessment['momentum']
        factors.append(momentum_factor)
        
        # Vulnerability capacity
        capacity_factor = min(1.0, profile.vulnerability_capacity / target_level)
        factors.append(capacity_factor)
        
        # Boundary flexibility
        if profile.boundaries:
            avg_flexibility = np.mean([b.flexibility for b in profile.boundaries])
            factors.append(avg_flexibility)
        else:
            factors.append(0.5)
        
        # Gap size (smaller gaps more likely)
        gap = target_level - profile.overall_intimacy
        gap_factor = 1.0 - (gap / 0.9)  # Normalize gap
        factors.append(max(0.1, gap_factor))
        
        # Calculate weighted probability
        weights = [0.25, 0.2, 0.2, 0.15, 0.2]
        probability = sum(f * w for f, w in zip(factors, weights))
        
        return min(0.95, max(0.05, probability))  # Cap between 5% and 95%
    
    def process_vulnerability_response(self, relationship_id: str,
                                     experience_id: str,
                                     response_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process response to vulnerability sharing"""
        if relationship_id not in self.intimacy_profiles:
            return {'error': 'No intimacy profile found'}
        
        profile = self.intimacy_profiles[relationship_id]
        
        # Find the vulnerability experience
        experience = None
        for event in profile.intimacy_history:
            if hasattr(event, 'experience_id') and event.experience_id == experience_id:
                experience = event
                break
        
        if not experience:
            return {'error': 'Vulnerability experience not found'}
        
        # Update experience with response
        experience.response_received = response_data.get('response', '')
        experience.outcome = response_data.get('outcome', 'neutral')
        
        # Calculate impacts
        if experience.outcome == 'positive':
            experience.trust_impact = 0.1 + experience.vulnerability_level * 0.2
            experience.intimacy_impact = 0.05 + experience.vulnerability_level * 0.15
        elif experience.outcome == 'negative':
            experience.trust_impact = -0.1 - experience.vulnerability_level * 0.1
            experience.intimacy_impact = -0.05 - experience.vulnerability_level * 0.1
        else:
            experience.trust_impact = 0.0
            experience.intimacy_impact = 0.02
        
        # Create intimacy event from vulnerability experience
        intimacy_event = IntimacyEvent(
            event_id=f"event_{datetime.now().timestamp()}",
            timestamp=datetime.now(),
            event_type='vulnerability_sharing',
            intimacy_type=IntimacyType.EMOTIONAL,
            description=f"Vulnerability shared and {experience.outcome}",
            intimacy_impact=experience.intimacy_impact,
            vulnerability_level=experience.vulnerability_level,
            reciprocated=experience.outcome == 'positive',
            outcomes={
                'trust_impact': experience.trust_impact,
                'experience_outcome': experience.outcome
            }
        )
        
        # Process the intimacy event
        result = self.process_intimacy_event(relationship_id, intimacy_event)
        
        # Update vulnerability capacity based on outcome
        self._update_vulnerability_capacity(intimacy_event, profile)
        
        return {
            'vulnerability_outcome': experience.outcome,
            'trust_impact': experience.trust_impact,
            'intimacy_impact': experience.intimacy_impact,
            'new_vulnerability_capacity': profile.vulnerability_capacity,
            'intimacy_update': result
        }
    
    def get_intimacy_insights(self, relationship_id: str) -> Dict[str, Any]:
        """Get deep insights about intimacy in the relationship"""
        if relationship_id not in self.intimacy_profiles:
            return {'error': 'No intimacy profile found'}
        
        profile = self.intimacy_profiles[relationship_id]
        
        # Analyze patterns
        patterns = self._analyze_intimacy_patterns(profile)
        
        # Identify turning points
        turning_points = self._identify_turning_points(profile)
        
        # Assess authenticity
        authenticity = self._assess_intimacy_authenticity(profile)
        
        # Project future trajectory
        future_projection = self._project_intimacy_future(profile)
        
        return {
            'current_state': {
                'level': profile.get_intimacy_level().value,
                'overall_score': profile.overall_intimacy,
                'strongest_type': profile.get_strongest_intimacy_type()
            },
            'patterns': patterns,
            'turning_points': turning_points,
            'authenticity_assessment': authenticity,
            'future_projection': future_projection,
            'key_insights': self._generate_key_insights(profile)
        }
    
    def _analyze_intimacy_patterns(self, profile: IntimacyProfile) -> Dict[str, Any]:
        """Analyze patterns in intimacy development"""
        if len(profile.intimacy_history) < 5:
            return {'status': 'insufficient_data'}
        
        # Analyze event patterns
        positive_events = [e for e in profile.intimacy_history if e.was_positive()]
        negative_events = [e for e in profile.intimacy_history if not e.was_positive()]
        
        # Check for cycles
        intimacy_levels = []
        for i in range(0, len(profile.intimacy_history), 5):
            subset = profile.intimacy_history[i:i+5]
            avg_impact = np.mean([e.intimacy_impact for e in subset])
            intimacy_levels.append(avg_impact)
        
        # Determine pattern type
        if len(intimacy_levels) > 2:
            variance = np.var(intimacy_levels)
            if variance < 0.01:
                pattern_type = 'stable'
            elif all(intimacy_levels[i] <= intimacy_levels[i+1] for i in range(len(intimacy_levels)-1)):
                pattern_type = 'consistently_growing'
            elif variance > 0.1:
                pattern_type = 'volatile'
            else:
                pattern_type = 'gradual_growth'
        else:
            pattern_type = 'emerging'
        
        return {
            'pattern_type': pattern_type,
            'positive_ratio': len(positive_events) / max(1, len(profile.intimacy_history)),
            'vulnerability_frequency': sum(1 for e in profile.intimacy_history if e.vulnerability_level > 0.5) / max(1, len(profile.intimacy_history)),
            'reciprocation_rate': sum(1 for e in profile.intimacy_history if e.reciprocated) / max(1, len(profile.intimacy_history))
        }
    
    def _identify_turning_points(self, profile: IntimacyProfile) -> List[Dict[str, Any]]:
        """Identify key turning points in intimacy development"""
        turning_points = []
        
        if len(profile.intimacy_history) < 3:
            return turning_points
        
        # Look for significant changes
        for i in range(1, len(profile.intimacy_history) - 1):
            prev_event = profile.intimacy_history[i-1]
            current_event = profile.intimacy_history[i]
            next_event = profile.intimacy_history[i+1]
            
            # Check for direction changes
            if (prev_event.intimacy_impact < 0 and current_event.intimacy_impact > 0.3) or \
               (prev_event.intimacy_impact > 0 and current_event.intimacy_impact < -0.3):
                turning_points.append({
                    'event': current_event,
                    'type': 'direction_change',
                    'significance': abs(current_event.intimacy_impact - prev_event.intimacy_impact)
                })
            
            # Check for breakthrough moments
            if current_event.vulnerability_level > 0.7 and current_event.was_positive():
                turning_points.append({
                    'event': current_event,
                    'type': 'vulnerability_breakthrough',
                    'significance': current_event.vulnerability_level
                })
        
        return sorted(turning_points, key=lambda x: x['significance'], reverse=True)[:5]
    
    def _assess_intimacy_authenticity(self, profile: IntimacyProfile) -> Dict[str, Any]:
        """Assess the authenticity of intimacy"""
        # Check for balanced growth
        dimension_levels = [d.current_level for d in profile.intimacy_dimensions.values()]
        balance_score = 1 - np.std(dimension_levels)
        
        # Check vulnerability patterns
        vulnerability_events = [e for e in profile.intimacy_history if e.vulnerability_level > 0.3]
        if vulnerability_events:
            avg_vulnerability = np.mean([e.vulnerability_level for e in vulnerability_events])
            vulnerability_consistency = 1 - np.std([e.vulnerability_level for e in vulnerability_events])
        else:
            avg_vulnerability = 0
            vulnerability_consistency = 0
        
        # Check reciprocation patterns
        reciprocation_rate = sum(1 for e in profile.intimacy_history if e.reciprocated) / max(1, len(profile.intimacy_history))
        
        # Calculate authenticity score
        authenticity_score = (balance_score * 0.3 + 
                            avg_vulnerability * 0.3 + 
                            vulnerability_consistency * 0.2 + 
                            reciprocation_rate * 0.2)
        
        return {
            'authenticity_score': authenticity_score,
            'balance_score': balance_score,
            'vulnerability_depth': avg_vulnerability,
            'consistency': vulnerability_consistency,
            'reciprocation': reciprocation_rate,
            'assessment': 'authentic' if authenticity_score > 0.6 else 'developing'
        }
    
    def _project_intimacy_future(self, profile: IntimacyProfile) -> Dict[str, Any]:
        """Project future intimacy trajectory"""
        # Get current growth trajectory
        growth_assessment = self._assess_intimacy_growth(profile)
        
        # Project levels at different time points
        current_level = profile.overall_intimacy
        growth_rate = growth_assessment['growth_rate']
        
        # Adjust for attachment style ceiling
        style_ceiling = self.attachment_styles[profile.attachment_style]['intimacy_readiness'] + 0.2
        
        projections = {}
        for months in [1, 3, 6, 12]:
            projected = current_level + (growth_rate * months * 4)  # 4 weeks per month
            projected = min(projected, style_ceiling)  # Cap at style ceiling
            projections[f'{months}_months'] = projected
        
        # Identify potential obstacles
        obstacles = []
        if profile.attachment_style == 'avoidant' and current_level > 0.5:
            obstacles.append('Avoidant attachment may create resistance')
        if growth_rate < 0.01:
            obstacles.append('Current growth rate is very slow')
        
        return {
            'current_trajectory': growth_assessment['trajectory'],
            'projections': projections,
            'potential_ceiling': style_ceiling,
            'obstacles': obstacles,
            'recommendations': self._generate_future_recommendations(profile, projections)
        }
    
    def _generate_key_insights(self, profile: IntimacyProfile) -> List[str]:
        """Generate key insights about the intimacy"""
        insights = []
        
        # Insight about overall level
        level = profile.get_intimacy_level()
        insights.append(f"Intimacy is at {level.value} level with score {profile.overall_intimacy:.2f}")
        
        # Insight about strongest dimension
        strongest = profile.get_strongest_intimacy_type()
        if strongest:
            insights.append(f"{strongest.value.capitalize()} intimacy is most developed")
        
        # Insight about growth
        growth = self._assess_intimacy_growth(profile)
        insights.append(f"Intimacy is {growth['trajectory'].replace('_', ' ')}")
        
        # Insight about attachment impact
        insights.append(f"{profile.attachment_style.capitalize()} attachment style influences intimacy capacity")
        
        # Insight about vulnerability
        if profile.vulnerability_capacity > 0.7:
            insights.append("High capacity for vulnerability supports deep connection")
        elif profile.vulnerability_capacity < 0.3:
            insights.append("Limited vulnerability capacity may restrict intimacy depth")
        
        return insights
    
    def _generate_future_recommendations(self, profile: IntimacyProfile,
                                       projections: Dict[str, float]) -> List[str]:
        """Generate recommendations for future intimacy development"""
        recommendations = []
        
        # Based on projections
        if projections['6_months'] < 0.5:
            recommendations.append('Focus on building foundational trust and connection')
        elif projections['6_months'] < 0.7:
            recommendations.append('Continue gradual deepening through shared experiences')
        else:
            recommendations.append('Maintain depth through consistent vulnerability and support')
        
        # Based on weakest dimension
        weakest_dim = min(
            profile.intimacy_dimensions.items(),
            key=lambda x: x[1].current_level
        )
        recommendations.append(f"Develop {weakest_dim[0].value} intimacy for better balance")
        
        # Based on attachment style
        if profile.attachment_style == 'avoidant':
            recommendations.append('Practice tolerating emotional closeness')
        elif profile.attachment_style == 'anxious':
            recommendations.append('Build secure base through consistent experiences')
        
        return recommendations
