#!/usr/bin/env python3
"""
Aphroditecho-Zero: Agent-Zero Integration with Aphrodite Engine

This module provides the integration layer between agent-zero's multi-agent
orchestration framework and Aphrodite Engine's inference architecture with
OpenCog cognitive systems and Deep Tree Echo membrane computing.

Key Features:
- Agent-zero multi-agent workbench integration
- HypergraphQL ECAN-aware atomspace bridge
- ASMOSES deep learning evolution integration  
- Yggdrasil decision-tree transformation
- Deep Tree Echo identity embedding
- AAR (Agent-Arena-Relation) orchestration core
"""

from .aphroditecho_zero import (
    AphroditechoZero,
    AphroditechoZeroConfig,
    AgentZeroAdapter,
)

from .opencog_bridge import (
    OpenCogAgentBridge,
    HypergraphQLAgentInterface,
)

from .deep_tree_echo_identity import (
    DeepTreeEchoIdentity,
    DTESNAgentWrapper,
)

from .aar_orchestrator import (
    AARAgentOrchestrator,
    AgentLifecycleManager,
    ArenaSimulator,
    RelationGraph,
)

__all__ = [
    'AphroditechoZero',
    'AphroditechoZeroConfig',
    'AgentZeroAdapter',
    'OpenCogAgentBridge',
    'HypergraphQLAgentInterface',
    'DeepTreeEchoIdentity',
    'DTESNAgentWrapper',
    'AARAgentOrchestrator',
    'AgentLifecycleManager',
    'ArenaSimulator',
    'RelationGraph',
]

__version__ = '0.1.0'
