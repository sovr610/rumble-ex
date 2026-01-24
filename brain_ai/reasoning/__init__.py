"""
Neuro-Symbolic Reasoning Module

System 2 deliberate reasoning with fuzzy logic.
"""

from .symbolic import (
    SymbolicReasoner,
    SymbolicConfig,
    FuzzyLogic,
    FuzzyLogicType,
    PredicateEncoder,
    RuleNetwork,
    KnowledgeBase,
    create_symbolic_reasoner,
)
from .system2 import (
    DualProcessReasoner,
    System2Config,
    System1Module,
    System2Module,
    MetacognitionModule,
    AdaptiveReasoner,
    create_dual_process_reasoner,
)

__all__ = [
    # Symbolic Reasoning
    'SymbolicReasoner',
    'SymbolicConfig',
    'FuzzyLogic',
    'FuzzyLogicType',
    'PredicateEncoder',
    'RuleNetwork',
    'KnowledgeBase',
    'create_symbolic_reasoner',
    # Dual-Process Reasoning
    'DualProcessReasoner',
    'System2Config',
    'System1Module',
    'System2Module',
    'MetacognitionModule',
    'AdaptiveReasoner',
    'create_dual_process_reasoner',
]
