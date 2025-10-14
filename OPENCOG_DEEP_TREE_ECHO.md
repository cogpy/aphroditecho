# OpenCog Deep Tree Echo Integration

## Overview

This integration brings OpenCog's cognitive architecture concepts to the Deep Tree Echo system, providing:

- **ECAN-aware AtomSpace**: Economic Attention Network-based attention allocation for cognitive resource management
- **HypergraphQL**: GraphQL-like query interface for flexible atomspace pattern matching and traversal
- **ASMOSES Integration**: Adaptive Symbolic MOSES for evolutionary program synthesis and neural architecture search
- **Hybrid Aphrodite-OpenCog**: Unified symbolic-neural architecture optimization

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                 OpenCog Deep Tree Echo System                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌──────────────────┐     ┌──────────────────┐                 │
│  │  OpenCog         │     │  HypergraphQL    │                 │
│  │  AtomSpace       │────▶│  Query Engine    │                 │
│  │  - ECAN          │     │  - Pattern Match │                 │
│  │  - Atoms/Links   │     │  - Traversal     │                 │
│  └──────────────────┘     └──────────────────┘                 │
│           │                        │                             │
│           │                        │                             │
│           ▼                        ▼                             │
│  ┌──────────────────────────────────────────┐                  │
│  │         Hybrid Integration                │                  │
│  │   ASMOSES ↔ Aphrodite Engine             │                  │
│  │   - Symbolic Program Evolution            │                  │
│  │   - Neural Architecture Search            │                  │
│  │   - Echo-Self Synchronization             │                  │
│  └──────────────────────────────────────────┘                  │
│                         │                                        │
│                         ▼                                        │
│  ┌──────────────────────────────────────────┐                  │
│  │      Deep Tree Echo Hypergraph            │                  │
│  │      Existing Echo Systems                │                  │
│  └──────────────────────────────────────────┘                  │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

## Components

### 1. OpenCog AtomSpace (`opencog_atomspace.py`)

The atomspace is the core knowledge representation structure, storing atoms (nodes and links) with associated truth values and attention values.

#### Key Features:

- **Atom Types**: ConceptNode, PredicateNode, SchemaNode, InheritanceLink, SimilarityLink, etc.
- **Truth Values**: Strength (probability/confidence) and confidence (amount of evidence)
- **Attention Values**: STI (Short-Term Importance), LTI (Long-Term Importance), VLTI (Very Long-Term Importance)
- **ECAN**: Economic Attention Network for resource allocation and attention spreading

#### Example Usage:

```python
from cognitive_architectures import OpenCogAtomSpace, AtomType, TruthValue

# Initialize atomspace with ECAN
atomspace = OpenCogAtomSpace(enable_ecan=True, sti_funds=100000.0)

# Add concept nodes
concept = atomspace.add_node(
    AtomType.CONCEPT_NODE,
    "deep-tree-echo",
    TruthValue(strength=0.95, confidence=0.9),
    initial_sti=100.0
)

# Add relationships
architecture = atomspace.add_node(
    AtomType.CONCEPT_NODE,
    "cognitive-architecture",
    TruthValue(0.9, 0.85)
)

link = atomspace.add_link(
    AtomType.INHERITANCE_LINK,
    [concept.id, architecture.id],
    TruthValue(0.95, 0.9)
)

# Spread attention through the graph
atomspace.spread_activation(concept.id, depth=2)

# Update ECAN (decay attention)
atomspace.update_ecan()

# Get atoms in attentional focus
focus = atomspace.get_attentional_focus(top_k=10)
```

### 2. HypergraphQL (`hypergraph_ql.py`)

A GraphQL-inspired query language for atomspace pattern matching and graph traversal.

#### Query Syntax:

```python
query = {
    'select': ['id', 'name', 'truth_value', 'attention'],
    'where': {
        'atom_type': 'ConceptNode',
        'name': {'contains': 'echo'},
        'truth_value.strength': {'gte': 0.7},
        'attention.sti': {'gt': 50}
    },
    'traverse': {
        'direction': 'outgoing',  # or 'incoming', 'both'
        'depth': 2,
        'filter': {'atom_type': 'InheritanceLink'}
    },
    'order_by': [{'field': 'attention.sti', 'direction': 'desc'}],
    'limit': 10
}
```

#### Example Usage:

```python
from cognitive_architectures import HypergraphQLEngine

engine = HypergraphQLEngine(atomspace)

# Execute query
result = engine.query(query)
print(f"Found {result.count} atoms in {result.query_time_ms:.2f}ms")

# Convenience methods
concepts = engine.find_concept("deep-tree-*", min_truth=0.7)
related = engine.find_related(concept.id, depth=2, direction='both')
top_attention = engine.find_by_attention(min_sti=100.0, limit=10)
```

### 3. ASMOSES Bridge (`asmoses_bridge.py`)

Adaptive Symbolic MOSES (Meta-Optimizing Semantic Evolutionary Search) for program synthesis and neural architecture evolution.

#### Key Features:

- **Program Trees**: Symbolic representation of programs and architectures
- **Evolutionary Operators**: Mutation, crossover, selection
- **Hybrid Optimization**: Combines symbolic and neural approaches
- **AtomSpace Integration**: Stores evolved programs in atomspace

#### Example Usage:

```python
from cognitive_architectures import (
    ASMOSESEvolution,
    ASMOSESConfig,
    ProgramTree,
    ProgramOperator
)

# Configure ASMOSES
config = ASMOSESConfig(
    population_size=100,
    max_generations=50,
    mutation_rate=0.1,
    crossover_rate=0.7
)

# Initialize with atomspace
asmoses = ASMOSESEvolution(config, atomspace)

# Define fitness function
def fitness_function(program_tree: ProgramTree) -> float:
    # Evaluate program/architecture
    # Return fitness score [0, 1]
    return evaluate_architecture(program_tree)

# Run evolution
best_program = await asmoses.evolve(
    fitness_function,
    target_fitness=0.95
)

print(f"Best fitness: {best_program.fitness:.4f}")
print(f"Complexity: {best_program.complexity}")
```

#### Hybrid Integration:

```python
from cognitive_architectures import HybridASMOSESAphroditeIntegration

hybrid = HybridASMOSESAphroditeIntegration(
    asmoses_config,
    atomspace,
    echo_self_evolution_engine
)

# Optimize architecture for specific task
result = await hybrid.optimize_architecture(
    task_specs={'task': 'classification', 'input_dim': 512},
    performance_metrics={'accuracy': 0.85, 'latency_ms': 10.0}
)

optimized_arch = result['architecture']
```

### 4. Main Integration (`opencog_deep_tree_echo.py`)

Unified interface combining all components with background processing and synchronization.

#### Example Usage:

```python
from cognitive_architectures import (
    OpenCogDeepTreeEcho,
    OpenCogDeepTreeEchoConfig
)

# Configure system
config = OpenCogDeepTreeEchoConfig(
    enable_ecan=True,
    asmoses_population_size=100,
    sync_interval_seconds=60.0,
    enable_attention_spreading=True
)

# Initialize system
system = OpenCogDeepTreeEcho(config)
await system.initialize()
await system.start()  # Start background tasks

# Add concepts
await system.add_concept(
    "neural-network-layer",
    truth_strength=0.9,
    initial_attention=80.0
)

# Create relationships
await system.add_relationship(
    "attention-mechanism",
    "transformer-architecture",
    relationship_type=AtomType.MEMBER_LINK
)

# Query with HypergraphQL
result = await system.query({
    'where': {'atom_type': 'ConceptNode'},
    'order_by': [{'field': 'attention.sti', 'direction': 'desc'}],
    'limit': 10
})

# Find related concepts
related = await system.find_related_concepts(
    "transformer-architecture",
    depth=2,
    min_truth=0.5
)

# Get attentional focus
focus = await system.get_attentional_focus(top_k=5)

# Spread attention
await system.spread_attention_from("key-concept", depth=2)

# Evolve architecture
optimized = await system.evolve_architecture(
    task_specs={'task': 'generation', 'context_length': 2048},
    performance_metrics={'perplexity': 15.2}
)

# Get statistics
stats = system.get_statistics()
print(f"Total atoms: {stats['atomspace']['total_atoms']}")
print(f"ECAN STI allocated: {stats['atomspace']['ecan']['sti_allocated']:.2f}")

# Cleanup
await system.stop()
```

## Integration with Existing Systems

### Deep Tree Echo Hypergraph

The OpenCog atomspace can import from and synchronize with the existing `DeepTreeEchoHypergraph`:

```python
from cognitive_architectures import DeepTreeEchoHypergraph

# Create existing hypergraph
hypergraph = DeepTreeEchoHypergraph()
# ... populate hypergraph ...

# Initialize OpenCog system with existing hypergraph
system = OpenCogDeepTreeEcho(config, existing_hypergraph=hypergraph)
await system.initialize()

# Hypernodes will be imported as ConceptNodes in atomspace
```

### Aphrodite Engine Integration

The ASMOSES bridge integrates with Aphrodite Engine for hybrid optimization:

1. **Architecture Search**: ASMOSES evolves program trees representing neural architectures
2. **Fitness Evaluation**: Architectures are evaluated using Aphrodite Engine
3. **Symbolic-Neural Fusion**: Best symbolic patterns combined with neural optimization
4. **Attention-Guided Search**: ECAN attention guides evolutionary search priorities

### Echo-Self Evolution Engine

Synchronize with the existing Echo-Self evolution engine:

```python
from echo_self.core.evolution_engine import EchoSelfEvolutionEngine

echo_self = EchoSelfEvolutionEngine(evolution_config)

hybrid = HybridASMOSESAphroditeIntegration(
    asmoses_config,
    atomspace,
    echo_self_evolution_engine=echo_self
)

# Synchronize populations
await hybrid.synchronize_with_echo_self()
```

## ECAN Attention Dynamics

The Economic Attention Network (ECAN) manages cognitive resources through:

### Attention Values

- **STI (Short-Term Importance)**: Immediate relevance, decays quickly
- **LTI (Long-Term Importance)**: Historical significance, decays slowly
- **VLTI (Very Long-Term Importance)**: Structural importance, stable

### Attention Allocation

```python
# Allocate attention to important concepts
system.ecan.allocate_attention(atom, sti_amount=100.0, lti_amount=50.0)

# Attention spreads through links
system.atomspace.spread_activation(source_atom.id, depth=2)

# Decay returns attention to available funds
system.atomspace.update_ecan()

# Get current attentional focus
focus = system.atomspace.get_attentional_focus(top_k=10)
```

### Economic Principles

- **Fixed Resources**: Total STI/LTI funds are conserved
- **Competitive Allocation**: Atoms compete for limited attention
- **Hebbian Spreading**: Attention flows along strong links
- **Decay Recycling**: Unused attention returns to the pool

## Pattern Matching

### Simple Patterns

```python
# Find all concept nodes with high truth values
result = engine.query({
    'where': {
        'atom_type': 'ConceptNode',
        'truth_value.strength': {'gte': 0.8}
    }
})
```

### Advanced Patterns with Variables

```python
# Find inheritance relationships
pattern = {
    'nodes': {
        '$x': {'atom_type': 'ConceptNode', 'name': 'deep-tree-echo'},
        '$y': {'atom_type': 'ConceptNode'}
    },
    'links': [
        {
            'atom_type': 'InheritanceLink',
            'outgoing': ['$x', '$y']
        }
    ]
}

bindings = engine.pattern_match(pattern)
for binding in bindings:
    print(f"{binding['$x'].name} inherits from {binding['$y'].name}")
```

### Graph Traversal

```python
# Find all concepts reachable within 3 hops
result = engine.query({
    'where': {'name': 'start-concept'},
    'traverse': {
        'direction': 'both',
        'depth': 3,
        'filter': {
            'truth_value.strength': {'gte': 0.5}
        }
    }
})
```

## Performance Considerations

### Caching

- HypergraphQL results are cached (configurable size)
- Pattern matching results are memoized
- ECAN computations use incremental updates

### Scaling

- AtomSpace supports millions of atoms
- ECAN attention focuses computation on important atoms
- Background tasks run asynchronously
- Query results support pagination

### Optimization Tips

1. **Use ECAN**: Enable ECAN to focus processing on important concepts
2. **Limit Query Depth**: Constrain traversal depth in queries
3. **Set Attention Thresholds**: Only spread attention from high-STI atoms
4. **Batch Operations**: Group atom additions and link creations
5. **Prune Low-Attention**: Periodically remove atoms with very low attention

## Testing

Run the comprehensive test suite:

```bash
pytest test_opencog_integration.py -v
```

Tests cover:
- AtomSpace operations (nodes, links, queries)
- ECAN attention allocation and spreading
- HypergraphQL queries and traversal
- ASMOSES evolution and program synthesis
- System integration and background tasks

## Future Enhancements

- [ ] Persistent AtomSpace storage (database backend)
- [ ] Distributed ECAN across multiple atomspaces
- [ ] PLN (Probabilistic Logic Networks) inference
- [ ] MOSES original algorithm implementation
- [ ] Real-time architecture deployment to Aphrodite
- [ ] Cognitive synergy metrics and monitoring
- [ ] Interactive visualization of attention dynamics
- [ ] Integration with additional OpenCog components

## References

- [OpenCog Framework](https://github.com/opencog/opencog)
- [ECAN Paper](https://wiki.opencog.org/w/ECAN)
- [MOSES Documentation](https://wiki.opencog.org/w/MOSES)
- [AtomSpace Design](https://wiki.opencog.org/w/AtomSpace)
- Deep Tree Echo: `ARCHITECTURE.md`, `DEEP_TREE_ECHO_ARCHITECTURE.md`
- Echo-Self: `echo_self/core/evolution_engine.py`

## Contributing

When extending the OpenCog integration:

1. Follow OpenCog conventions for atom types and truth values
2. Maintain ECAN economic principles (conserve attention resources)
3. Add comprehensive tests for new features
4. Document query patterns and usage examples
5. Consider performance impact of attention spreading
6. Ensure compatibility with existing Deep Tree Echo components

## License

This integration follows the same license as the parent Aphrodite Engine project.
