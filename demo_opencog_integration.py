#!/usr/bin/env python3
"""
Demo script for OpenCog Deep Tree Echo Integration

Demonstrates the key features of the OpenCog integration:
- ECAN-aware atomspace with attention allocation
- HypergraphQL queries
- ASMOSES evolutionary architecture search
- Hybrid symbolic-neural optimization
"""

import asyncio
import sys
from cognitive_architectures import (
    OpenCogDeepTreeEcho,
    OpenCogDeepTreeEchoConfig,
    AtomType,
    demo_opencog_deep_tree_echo
)


async def main():
    """Run the comprehensive demo."""
    print("=" * 70)
    print("OpenCog Deep Tree Echo Integration Demo")
    print("=" * 70)
    print()
    
    # Run the built-in demo
    await demo_opencog_deep_tree_echo()
    
    print("\n" + "=" * 70)
    print("Additional Integration Examples")
    print("=" * 70)
    print()
    
    # Create a new system for additional examples
    config = OpenCogDeepTreeEchoConfig(
        enable_ecan=True,
        asmoses_population_size=20,
        asmoses_max_generations=5
    )
    
    system = OpenCogDeepTreeEcho(config)
    await system.initialize()
    await system.start()
    
    print("\n1. Building a Knowledge Graph")
    print("-" * 50)
    
    # Add AI/ML concepts
    concepts = [
        ("machine-learning", 0.95, 100.0),
        ("deep-learning", 0.92, 90.0),
        ("neural-networks", 0.90, 85.0),
        ("transformers", 0.88, 80.0),
        ("attention-mechanism", 0.85, 75.0),
        ("language-models", 0.87, 78.0),
    ]
    
    for name, truth, attention in concepts:
        await system.add_concept(name, truth_strength=truth, initial_attention=attention)
        print(f"  Added: {name} (truth={truth:.2f}, attention={attention:.1f})")
    
    # Create relationships
    print("\n2. Creating Semantic Relationships")
    print("-" * 50)
    
    relationships = [
        ("deep-learning", "machine-learning", AtomType.INHERITANCE_LINK),
        ("neural-networks", "deep-learning", AtomType.MEMBER_LINK),
        ("transformers", "neural-networks", AtomType.INHERITANCE_LINK),
        ("attention-mechanism", "transformers", AtomType.MEMBER_LINK),
        ("language-models", "transformers", AtomType.SIMILARITY_LINK),
    ]
    
    for source, target, link_type in relationships:
        await system.add_relationship(source, target, link_type)
        print(f"  {source} --[{link_type.value}]--> {target}")
    
    # Query the knowledge graph
    print("\n3. Querying with HypergraphQL")
    print("-" * 50)
    
    # Find all concepts related to machine learning
    result = await system.query({
        'where': {
            'atom_type': 'ConceptNode',
            'truth_value.strength': {'gte': 0.85}
        },
        'order_by': [{'field': 'attention.sti', 'direction': 'desc'}],
        'limit': 5
    })
    
    print(f"  Found {result.count} high-confidence concepts:")
    for atom in result.atoms[:5]:
        print(f"    - {atom.name}: truth={atom.truth_value.strength:.3f}, "
              f"attention={atom.attention.sti:.1f}")
    
    # Find concepts related to transformers
    print("\n4. Graph Traversal")
    print("-" * 50)
    
    related = await system.find_related_concepts("transformers", depth=2)
    print(f"  Found {len(related)} concepts related to 'transformers':")
    for atom in related[:5]:
        if atom.name:
            print(f"    - {atom.name}")
    
    # Demonstrate attention spreading
    print("\n5. ECAN Attention Dynamics")
    print("-" * 50)
    
    print("  Before attention spreading:")
    focus_before = await system.get_attentional_focus(top_k=3)
    for i, atom in enumerate(focus_before):
        print(f"    {i+1}. {atom.name}: STI={atom.attention.sti:.2f}")
    
    # Spread attention from a key concept
    await system.spread_attention_from("transformers", depth=2)
    
    print("\n  After spreading attention from 'transformers':")
    focus_after = await system.get_attentional_focus(top_k=3)
    for i, atom in enumerate(focus_after):
        print(f"    {i+1}. {atom.name}: STI={atom.attention.sti:.2f}")
    
    # Pattern matching example
    print("\n6. Advanced Pattern Matching")
    print("-" * 50)
    
    pattern = {
        'nodes': {
            '$parent': {'atom_type': 'ConceptNode', 'name': 'machine-learning'},
            '$child': {'atom_type': 'ConceptNode'}
        },
        'links': [
            {
                'atom_type': 'InheritanceLink',
                'outgoing': ['$child', '$parent']
            }
        ]
    }
    
    matches = await system.pattern_match(pattern)
    print(f"  Found {len(matches)} inheritance relationships with 'machine-learning'")
    for match in matches:
        if '$child' in match and match['$child'].name:
            print(f"    - {match['$child'].name} inherits from machine-learning")
    
    # System statistics
    print("\n7. System Statistics")
    print("-" * 50)
    
    stats = system.get_statistics()
    print(f"  Total atoms: {stats['atomspace']['total_atoms']}")
    print(f"  Nodes: {stats['atomspace']['nodes']}")
    print(f"  Links: {stats['atomspace']['links']}")
    
    if 'ecan' in stats['atomspace']:
        ecan = stats['atomspace']['ecan']
        print(f"  STI allocated: {ecan['sti_allocated']:.2f}")
        print(f"  STI available: {ecan['sti_available']:.2f}")
        print(f"  Attentional focus size: {ecan['attentional_focus_size']}")
    
    print(f"  ASMOSES generation: {stats['asmoses']['generation']}")
    print(f"  ASMOSES population: {stats['asmoses']['population_size']}")
    
    # Cleanup
    await system.stop()
    
    print("\n" + "=" * 70)
    print("Demo completed successfully!")
    print("=" * 70)
    print()
    print("Key Takeaways:")
    print("  ✓ OpenCog atomspace integrated with Deep Tree Echo")
    print("  ✓ ECAN attention allocation managing cognitive resources")
    print("  ✓ HypergraphQL providing flexible queries")
    print("  ✓ ASMOSES ready for evolutionary architecture search")
    print("  ✓ Hybrid symbolic-neural optimization enabled")
    print()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n\nError running demo: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
