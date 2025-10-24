"""
Cognitive Architectures Module

Provides OpenCog-based Deep Tree Echo integration with:
- ECAN-aware atomspace for attention allocation
- HypergraphQL for flexible querying
- ASMOSES for evolutionary architecture optimization
- Hybrid integration with Aphrodite Engine
"""

from .opencog_atomspace import (
    OpenCogAtomSpace,
    Atom,
    AtomType,
    TruthValue,
    AttentionValue,
    ECANAttentionAllocator,
    initialize_deep_tree_echo_atomspace
)

from .hypergraph_ql import (
    HypergraphQLEngine,
    HypergraphQLParser,
    QueryResult,
    QueryOperator
)

from .asmoses_bridge import (
    ASMOSESEvolution,
    ASMOSESConfig,
    ASMOSESPopulation,
    EvolvedProgram,
    ProgramTree,
    ProgramOperator,
    HybridASMOSESAphroditeIntegration
)

from .opencog_deep_tree_echo import (
    OpenCogDeepTreeEcho,
    OpenCogDeepTreeEchoConfig,
    initialize_deep_tree_echo_atomspace_concepts,
    demo_opencog_deep_tree_echo
)

from .echoself_hypergraph_data import (
    DeepTreeEchoHypergraph,
    IdentityRole,
    MemoryType,
    HyperedgeType
)

__all__ = [
    # AtomSpace
    'OpenCogAtomSpace',
    'Atom',
    'AtomType',
    'TruthValue',
    'AttentionValue',
    'ECANAttentionAllocator',
    'initialize_deep_tree_echo_atomspace',
    
    # HypergraphQL
    'HypergraphQLEngine',
    'HypergraphQLParser',
    'QueryResult',
    'QueryOperator',
    
    # ASMOSES
    'ASMOSESEvolution',
    'ASMOSESConfig',
    'ASMOSESPopulation',
    'EvolvedProgram',
    'ProgramTree',
    'ProgramOperator',
    'HybridASMOSESAphroditeIntegration',
    
    # Main Integration
    'OpenCogDeepTreeEcho',
    'OpenCogDeepTreeEchoConfig',
    'initialize_deep_tree_echo_atomspace_concepts',
    'demo_opencog_deep_tree_echo',
    
    # Existing Hypergraph
    'DeepTreeEchoHypergraph',
    'IdentityRole',
    'MemoryType',
    'HyperedgeType',
]
