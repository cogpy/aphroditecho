# OpenCog Deep Tree Echo - Quick Start Guide

## Installation

No additional dependencies required beyond the standard Aphrodite Engine setup. The OpenCog integration uses only:
- `numpy` (for entropy calculations)
- Standard library modules

## Basic Usage

### 1. Initialize the System

```python
from cognitive_architectures import (
    OpenCogDeepTreeEcho,
    OpenCogDeepTreeEchoConfig,
    AtomType
)

# Create configuration
config = OpenCogDeepTreeEchoConfig(
    enable_ecan=True,              # Enable attention allocation
    sti_funds=100000.0,            # Total STI budget
    lti_funds=100000.0,            # Total LTI budget
    asmoses_population_size=100,   # Evolution population size
    enable_attention_spreading=True # Auto-spread attention
)

# Initialize system
system = OpenCogDeepTreeEcho(config)
await system.initialize()
await system.start()  # Start background tasks
```

### 2. Add Concepts and Relationships

```python
# Add concepts with attention
await system.add_concept(
    "neural-network",
    truth_strength=0.9,      # Confidence in concept [0, 1]
    truth_confidence=0.8,    # Amount of evidence [0, 1]
    initial_attention=80.0   # Initial STI value
)

await system.add_concept("deep-learning", truth_strength=0.95, initial_attention=100.0)

# Create relationship
await system.add_relationship(
    "neural-network",
    "deep-learning",
    relationship_type=AtomType.INHERITANCE_LINK,
    truth_strength=0.9
)
```

### 3. Query with HypergraphQL

```python
# Simple query
result = await system.query({
    'where': {
        'atom_type': 'ConceptNode',
        'truth_value.strength': {'gte': 0.8}
    },
    'limit': 10
})

print(f"Found {result.count} concepts")
for atom in result.atoms:
    print(f"  - {atom.name}: {atom.truth_value.strength:.2f}")

# Query with traversal
result = await system.query({
    'where': {'name': 'neural-network'},
    'traverse': {
        'direction': 'both',  # 'outgoing', 'incoming', or 'both'
        'depth': 2,
        'filter': {'truth_value.strength': {'gte': 0.7}}
    }
})

# Query with ordering
result = await system.query({
    'where': {'atom_type': 'ConceptNode'},
    'order_by': [
        {'field': 'attention.sti', 'direction': 'desc'}
    ],
    'limit': 5
})
```

### 4. Work with Attention

```python
# Get current attentional focus
focus = await system.get_attentional_focus(top_k=5)
for atom in focus:
    print(f"{atom.name}: STI={atom.attention.sti:.2f}")

# Spread attention from a concept
await system.spread_attention_from("neural-network", depth=2)

# Find high-attention concepts
from cognitive_architectures import HypergraphQLEngine
engine = HypergraphQLEngine(system.atomspace)
top_atoms = engine.find_by_attention(min_sti=50.0, limit=10)
```

### 5. Evolve Architectures with ASMOSES

```python
# Define optimization task
task_specs = {
    'task': 'classification',
    'input_dim': 512,
    'output_dim': 10
}

performance_metrics = {
    'accuracy': 0.85,
    'latency_ms': 10.0
}

# Run evolution
result = await system.evolve_architecture(
    task_specs,
    performance_metrics
)

print(f"Optimized architecture:")
print(f"  Fitness: {result['fitness']:.4f}")
print(f"  Complexity: {result['complexity']}")
print(f"  Architecture: {result['architecture']}")
```

### 6. Pattern Matching

```python
# Find inheritance relationships
pattern = {
    'nodes': {
        '$child': {'atom_type': 'ConceptNode'},
        '$parent': {'atom_type': 'ConceptNode', 'name': 'deep-learning'}
    },
    'links': [
        {
            'atom_type': 'InheritanceLink',
            'outgoing': ['$child', '$parent']
        }
    ]
}

matches = await system.pattern_match(pattern)
for match in matches:
    print(f"{match['$child'].name} inherits from {match['$parent'].name}")
```

### 7. Get System Statistics

```python
stats = system.get_statistics()

print(f"Atomspace:")
print(f"  Total atoms: {stats['atomspace']['total_atoms']}")
print(f"  Nodes: {stats['atomspace']['nodes']}")
print(f"  Links: {stats['atomspace']['links']}")

print(f"\nECAN:")
print(f"  STI allocated: {stats['atomspace']['ecan']['sti_allocated']:.2f}")
print(f"  STI available: {stats['atomspace']['ecan']['sti_available']:.2f}")

print(f"\nASMOSES:")
print(f"  Generation: {stats['asmoses']['generation']}")
print(f"  Population size: {stats['asmoses']['population_size']}")
```

### 8. Cleanup

```python
# Stop background tasks
await system.stop()
```

## Common Patterns

### Knowledge Graph Construction

```python
# Add domain concepts
concepts = [
    ("ai", 1.0, 100.0),
    ("machine-learning", 0.95, 90.0),
    ("neural-networks", 0.90, 85.0),
]

for name, truth, attention in concepts:
    await system.add_concept(name, truth_strength=truth, initial_attention=attention)

# Create taxonomy
await system.add_relationship("machine-learning", "ai", AtomType.INHERITANCE_LINK)
await system.add_relationship("neural-networks", "machine-learning", AtomType.INHERITANCE_LINK)
```

### Attention-Guided Processing

```python
# Process concepts in order of attention
focus = await system.get_attentional_focus(top_k=10)
for atom in focus:
    # Process high-attention concepts first
    result = await process_concept(atom)
    
    # Boost attention if successful
    if result.success:
        system.atomspace.ecan.allocate_attention(atom, sti_amount=20.0)
```

### Multi-Hop Reasoning

```python
# Find concepts connected through multiple relationships
result = await system.query({
    'where': {'name': 'start-concept'},
    'traverse': {
        'direction': 'both',
        'depth': 3,
        'filter': {
            'atom_type': 'ConceptNode',
            'truth_value.strength': {'gte': 0.7}
        }
    }
})
```

### Architecture Search with Constraints

```python
# Define fitness function with constraints
async def constrained_fitness(program_tree):
    arch = convert_tree_to_architecture(program_tree)
    
    # Evaluate performance
    performance = await evaluate_architecture(arch)
    
    # Apply constraints
    if arch['parameter_count'] > 1e9:  # Max 1B parameters
        performance *= 0.5
    
    if arch['latency_ms'] > 100:  # Max 100ms latency
        performance *= 0.7
    
    return performance

# Run constrained evolution
best = await system.asmoses.evolve(constrained_fitness, target_fitness=0.9)
```

## Tips and Best Practices

1. **Attention Management**: Use ECAN to focus computation on important concepts
2. **Query Optimization**: Use specific filters and limit depth to avoid expensive traversals
3. **Background Tasks**: Start system with `await system.start()` to enable automatic ECAN updates
4. **Truth Values**: Higher confidence values indicate more reliable information
5. **Pattern Matching**: Use specific patterns to reduce search space
6. **Evolution**: Start with small populations for faster iteration, increase for final optimization

## Running the Demo

```bash
python demo_opencog_integration.py
```

This runs a comprehensive demonstration of all features.

## Testing

```bash
pytest test_opencog_integration.py -v
```

All 30 tests should pass.

## Next Steps

- Read full documentation: `OPENCOG_DEEP_TREE_ECHO.md`
- Explore atomspace patterns: `cognitive_architectures/opencog_atomspace.py`
- Try HypergraphQL queries: `cognitive_architectures/hypergraph_ql.py`
- Experiment with ASMOSES: `cognitive_architectures/asmoses_bridge.py`
- Integrate with existing systems: `cognitive_architectures/opencog_deep_tree_echo.py`

## Support

For issues or questions:
1. Check the main documentation
2. Review test cases for examples
3. Run the demo script
4. Examine the architecture diagrams in `ARCHITECTURE.md`
