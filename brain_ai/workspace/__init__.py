"""
Global Workspace Module

Attention-based multi-modal integration and working memory.
"""

from .working_memory import (
    WorkingMemory,
    WorkingMemoryConfig,
    GRUWorkingMemory,
    create_working_memory,
)
from .global_workspace import (
    GlobalWorkspace,
    GlobalWorkspaceConfig,
    GlobalWorkspaceWithHTM,
    AttentionCompetition,
    InformationBroadcast,
    create_global_workspace,
    # Improved Selection-Broadcast (2025)
    SelectionBroadcastConfig,
    IterativeCompetition,
    RefinedBroadcast,
    SelectionBroadcastWorkspace,
    create_selection_broadcast_workspace,
)

__all__ = [
    # Working Memory
    'WorkingMemory',
    'WorkingMemoryConfig',
    'GRUWorkingMemory',
    'create_working_memory',
    # Global Workspace
    'GlobalWorkspace',
    'GlobalWorkspaceConfig',
    'GlobalWorkspaceWithHTM',
    'AttentionCompetition',
    'InformationBroadcast',
    'create_global_workspace',
    # Improved Selection-Broadcast (2025)
    'SelectionBroadcastConfig',
    'IterativeCompetition',
    'RefinedBroadcast',
    'SelectionBroadcastWorkspace',
    'create_selection_broadcast_workspace',
]
