"""
Meta-Learning Module

Plasticity control, MAML, and eligibility traces.
"""

from .neuromodulation import (
    NeuromodulatoryGate,
    NeuromodulationConfig,
    PlasticityController,
    DopamineSystem,
    AcetylcholineSystem,
    NorepinephrineSystem,
    SerotoninSystem,
    create_neuromodulatory_gate,
)
from .maml import (
    MAML,
    MAMLConfig,
    FOMAML,
    Reptile,
    MetaLearner,
    InnerLoopOptimizer,
    create_meta_learner,
)
from .eligibility import (
    EligibilityTrace,
    EligibilityConfig,
    EligibilityNetwork,
    EligibilityMLP,
    OnlineLearner,
    TemporalDifferenceTrace,
    create_eligibility_network,
)

__all__ = [
    # Neuromodulation
    'NeuromodulatoryGate',
    'NeuromodulationConfig',
    'PlasticityController',
    'DopamineSystem',
    'AcetylcholineSystem',
    'NorepinephrineSystem',
    'SerotoninSystem',
    'create_neuromodulatory_gate',
    # MAML
    'MAML',
    'MAMLConfig',
    'FOMAML',
    'Reptile',
    'MetaLearner',
    'InnerLoopOptimizer',
    'create_meta_learner',
    # Eligibility Traces
    'EligibilityTrace',
    'EligibilityConfig',
    'EligibilityNetwork',
    'EligibilityMLP',
    'OnlineLearner',
    'TemporalDifferenceTrace',
    'create_eligibility_network',
]
