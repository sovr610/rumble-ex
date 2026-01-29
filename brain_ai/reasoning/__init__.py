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
    # Logic Tensor Networks (2025)
    RealLogic,
    LTNPredicate,
    LTNFunction,
    LTNConstant,
    LTNVariable,
    LogicTensorNetwork,
    LTNSatisfactionAggregator,
    ltn_loss,
    create_ltn,
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
    # Logic Tensor Networks (2025)
    'RealLogic',
    'LTNPredicate',
    'LTNFunction',
    'LTNConstant',
    'LTNVariable',
    'LogicTensorNetwork',
    'LTNSatisfactionAggregator',
    'ltn_loss',
    'create_ltn',
    # Dual-Process Reasoning
    'DualProcessReasoner',
    'System2Config',
    'System1Module',
    'System2Module',
    'MetacognitionModule',
    'AdaptiveReasoner',
    'create_dual_process_reasoner',
]
