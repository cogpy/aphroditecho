#!/usr/bin/env python3
"""
OpenCog Deep Tree Echo Integration

Main integration module connecting OpenCog atomspace, HypergraphQL, ASMOSES,
and Aphrodite Engine for hybrid symbolic-neural cognitive architecture.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

from .opencog_atomspace import OpenCogAtomSpace, Atom, AtomType, TruthValue, initialize_deep_tree_echo_atomspace
from .hypergraph_ql import HypergraphQLEngine, QueryResult
from .asmoses_bridge import (
    ASMOSESEvolution,
    ASMOSESConfig,
    HybridASMOSESAphroditeIntegration,
    EvolvedProgram,
    ProgramTree
)
from .echoself_hypergraph_data import DeepTreeEchoHypergraph

logger = logging.getLogger(__name__)


@dataclass
class OpenCogDeepTreeEchoConfig:
    """Configuration for OpenCog Deep Tree Echo integration."""
    # AtomSpace settings
    enable_ecan: bool = True
    sti_funds: float = 100000.0
    lti_funds: float = 100000.0
    
    # ASMOSES settings
    asmoses_population_size: int = 100
    asmoses_max_generations: int = 50
    asmoses_mutation_rate: float = 0.1
    
    # HypergraphQL settings
    query_cache_size: int = 1000
    max_query_depth: int = 5
    
    # Integration settings
    sync_interval_seconds: float = 60.0
    enable_attention_spreading: bool = True
    enable_atomspace_persistence: bool = True


class OpenCogDeepTreeEcho:
    """
    Main integration class for OpenCog-based Deep Tree Echo.
    
    Provides unified interface for:
    - ECAN-aware atomspace cognitive processing
    - HypergraphQL querying and pattern matching
    - ASMOSES evolutionary architecture optimization
    - Hybrid integration with Aphrodite Engine
    """
    
    def __init__(
        self,
        config: Optional[OpenCogDeepTreeEchoConfig] = None,
        existing_hypergraph: Optional[DeepTreeEchoHypergraph] = None
    ):
        self.config = config or OpenCogDeepTreeEchoConfig()
        
        # Initialize OpenCog atomspace
        self.atomspace = OpenCogAtomSpace(
            enable_ecan=self.config.enable_ecan,
            sti_funds=self.config.sti_funds,
            lti_funds=self.config.lti_funds
        )
        
        # Initialize HypergraphQL engine
        self.hypergraph_ql = HypergraphQLEngine(self.atomspace)
        
        # Initialize ASMOSES
        asmoses_config = ASMOSESConfig(
            population_size=self.config.asmoses_population_size,
            max_generations=self.config.asmoses_max_generations,
            mutation_rate=self.config.asmoses_mutation_rate,
            enable_atomspace_storage=True
        )
        self.asmoses = ASMOSESEvolution(asmoses_config, self.atomspace)
        
        # Hybrid integration
        self.hybrid_integration = HybridASMOSESAphroditeIntegration(
            asmoses_config,
            self.atomspace
        )
        
        # Link to existing hypergraph if provided
        self.external_hypergraph = existing_hypergraph
        
        # Background tasks
        self.background_tasks: List[asyncio.Task] = []
        self.is_running = False
        
        logger.info("OpenCog Deep Tree Echo integration initialized")
    
    async def initialize(self):
        """Initialize the system with foundational cognitive structures."""
        # Initialize atomspace with Deep Tree Echo concepts
        initialize_deep_tree_echo_atomspace_concepts(self.atomspace)
        
        # Import from external hypergraph if available
        if self.external_hypergraph:
            await self._import_from_external_hypergraph()
        
        logger.info("OpenCog Deep Tree Echo system initialized")
    
    async def start(self):
        """Start background processing tasks."""
        if self.is_running:
            logger.warning("System already running")
            return
        
        self.is_running = True
        
        # Start ECAN update loop
        if self.config.enable_ecan:
            task = asyncio.create_task(self._ecan_update_loop())
            self.background_tasks.append(task)
        
        # Start attention spreading if enabled
        if self.config.enable_attention_spreading:
            task = asyncio.create_task(self._attention_spreading_loop())
            self.background_tasks.append(task)
        
        logger.info("OpenCog Deep Tree Echo background tasks started")
    
    async def stop(self):
        """Stop background processing tasks."""
        self.is_running = False
        
        # Cancel all background tasks
        for task in self.background_tasks:
            task.cancel()
        
        await asyncio.gather(*self.background_tasks, return_exceptions=True)
        self.background_tasks.clear()
        
        logger.info("OpenCog Deep Tree Echo stopped")
    
    async def query(self, query: Dict[str, Any]) -> QueryResult:
        """Execute HypergraphQL query on the atomspace."""
        return self.hypergraph_ql.query(query)
    
    async def add_concept(
        self,
        name: str,
        truth_strength: float = 0.8,
        truth_confidence: float = 0.7,
        initial_attention: float = 50.0,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Atom:
        """
        Add a concept to the atomspace with optional metadata.
        
        Args:
            name: Concept name
            truth_strength: Truth value strength [0, 1]
            truth_confidence: Truth value confidence [0, 1]
            initial_attention: Initial STI for ECAN
            metadata: Optional metadata dictionary
        
        Returns:
            Created atom
        """
        atom = self.atomspace.add_node(
            AtomType.CONCEPT_NODE,
            name,
            TruthValue(truth_strength, truth_confidence),
            initial_sti=initial_attention
        )
        
        if metadata:
            atom.metadata.update(metadata)
        
        logger.debug(f"Added concept: {name} with attention {initial_attention}")
        return atom
    
    async def add_relationship(
        self,
        source_name: str,
        target_name: str,
        relationship_type: AtomType = AtomType.SIMILARITY_LINK,
        truth_strength: float = 0.8,
        truth_confidence: float = 0.7
    ) -> Optional[Atom]:
        """
        Add a relationship between two concepts.
        
        Args:
            source_name: Source concept name
            target_name: Target concept name
            relationship_type: Type of relationship link
            truth_strength: Truth value strength
            truth_confidence: Truth value confidence
        
        Returns:
            Created link atom or None if concepts not found
        """
        # Find source and target atoms
        source_atoms = self.atomspace.get_atoms_by_name(source_name)
        target_atoms = self.atomspace.get_atoms_by_name(target_name)
        
        if not source_atoms or not target_atoms:
            logger.warning(
                f"Cannot create relationship: source or target not found "
                f"({source_name} -> {target_name})"
            )
            return None
        
        source = source_atoms[0]
        target = target_atoms[0]
        
        link = self.atomspace.add_link(
            relationship_type,
            [source.id, target.id],
            TruthValue(truth_strength, truth_confidence)
        )
        
        logger.debug(f"Added relationship: {source_name} -> {target_name}")
        return link
    
    async def find_related_concepts(
        self,
        concept_name: str,
        depth: int = 2,
        min_truth: float = 0.5
    ) -> List[Atom]:
        """
        Find concepts related to given concept.
        
        Args:
            concept_name: Name of the concept
            depth: Maximum traversal depth
            min_truth: Minimum truth value threshold
        
        Returns:
            List of related atoms
        """
        # Find the concept
        atoms = self.atomspace.get_atoms_by_name(concept_name)
        if not atoms:
            logger.warning(f"Concept not found: {concept_name}")
            return []
        
        concept_atom = atoms[0]
        
        # Query for related atoms
        query = {
            'where': {'id': concept_atom.id},
            'traverse': {
                'direction': 'both',
                'depth': depth,
                'filter': {
                    'truth_value.strength': {'gte': min_truth}
                }
            }
        }
        
        result = await self.query(query)
        return result.atoms
    
    async def get_attentional_focus(self, top_k: int = 10) -> List[Atom]:
        """Get atoms currently in attentional focus (highest STI)."""
        return self.atomspace.get_attentional_focus(top_k)
    
    async def spread_attention_from(
        self,
        concept_name: str,
        depth: int = 2
    ):
        """Spread attention from a specific concept through the graph."""
        atoms = self.atomspace.get_atoms_by_name(concept_name)
        if not atoms:
            logger.warning(f"Concept not found for attention spreading: {concept_name}")
            return
        
        self.atomspace.spread_activation(atoms[0].id, depth)
        logger.debug(f"Spread attention from {concept_name} to depth {depth}")
    
    async def evolve_architecture(
        self,
        task_specs: Dict[str, Any],
        performance_metrics: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Evolve neural architecture using ASMOSES.
        
        Args:
            task_specs: Task specifications for optimization
            performance_metrics: Current performance metrics
        
        Returns:
            Optimized architecture specification
        """
        metrics = performance_metrics or {}
        
        result = await self.hybrid_integration.optimize_architecture(
            task_specs,
            metrics
        )
        
        logger.info(
            f"Architecture evolution completed: "
            f"fitness={result['fitness']:.4f}, "
            f"complexity={result['complexity']}"
        )
        
        return result
    
    async def pattern_match(
        self,
        pattern: Dict[str, Any],
        variables: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Atom]]:
        """
        Advanced pattern matching with variable binding.
        
        Args:
            pattern: Pattern specification
            variables: Variable bindings
        
        Returns:
            List of variable bindings matching the pattern
        """
        return self.hypergraph_ql.pattern_match(pattern, variables)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive system statistics."""
        stats = {
            'atomspace': self.atomspace.get_statistics(),
            'asmoses': {
                'generation': self.asmoses.population.generation,
                'population_size': len(self.asmoses.population.programs),
                'evolution_history_length': len(self.asmoses.evolution_history)
            },
            'system': {
                'is_running': self.is_running,
                'background_tasks': len(self.background_tasks)
            }
        }
        
        return stats
    
    async def _ecan_update_loop(self):
        """Background loop for ECAN updates."""
        while self.is_running:
            try:
                self.atomspace.update_ecan()
                await asyncio.sleep(self.config.sync_interval_seconds)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in ECAN update loop: {e}")
                await asyncio.sleep(5.0)
    
    async def _attention_spreading_loop(self):
        """Background loop for automatic attention spreading."""
        while self.is_running:
            try:
                # Get top atoms in focus
                focus_atoms = self.atomspace.get_attentional_focus(top_k=5)
                
                # Spread attention from each
                for atom in focus_atoms:
                    if atom.attention.sti > 100:
                        self.atomspace.spread_activation(atom.id, depth=2)
                
                await asyncio.sleep(self.config.sync_interval_seconds * 2)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in attention spreading loop: {e}")
                await asyncio.sleep(5.0)
    
    async def _import_from_external_hypergraph(self):
        """Import structures from external DeepTreeEcho hypergraph."""
        if not self.external_hypergraph:
            return
        
        logger.info("Importing from external hypergraph...")
        
        # Import hypernodes as concepts
        for node_id, hypernode in self.external_hypergraph.hypernodes.items():
            concept_name = f"hypernode-{hypernode.current_role.value}-{node_id[:8]}"
            
            await self.add_concept(
                name=concept_name,
                truth_strength=hypernode.activation_level,
                truth_confidence=0.8,
                initial_attention=hypernode.activation_level * 100,
                metadata={
                    'identity_seed': hypernode.identity_seed,
                    'role': hypernode.current_role.value,
                    'entropy_trace': hypernode.entropy_trace[-10:] if hypernode.entropy_trace else []
                }
            )
        
        logger.info(
            f"Imported {len(self.external_hypergraph.hypernodes)} hypernodes "
            "from external hypergraph"
        )


def initialize_deep_tree_echo_atomspace_concepts(atomspace: OpenCogAtomSpace):
    """Initialize atomspace with comprehensive Deep Tree Echo concepts."""
    # Call the base initialization
    initialize_deep_tree_echo_atomspace()
    
    # Add additional Deep Tree Echo specific concepts
    concepts = [
        ("4e-embodied-ai", 0.95),
        ("sensory-motor-mapping", 0.9),
        ("proprioceptive-feedback", 0.9),
        ("membrane-computing", 0.85),
        ("echo-state-networks", 0.85),
        ("agent-arena-relation", 0.9),
        ("cognitive-synergy", 0.85),
        ("attention-allocation", 0.8),
        ("pattern-recognition", 0.8),
        ("symbolic-reasoning", 0.85),
        ("neural-architecture-search", 0.8)
    ]
    
    for concept_name, truth_strength in concepts:
        atomspace.add_node(
            AtomType.CONCEPT_NODE,
            concept_name,
            TruthValue(truth_strength, 0.8),
            initial_sti=truth_strength * 80
        )
    
    # Create hierarchical relationships
    relationships = [
        ("4e-embodied-ai", "deep-tree-echo", AtomType.MEMBER_LINK),
        ("sensory-motor-mapping", "4e-embodied-ai", AtomType.MEMBER_LINK),
        ("proprioceptive-feedback", "4e-embodied-ai", AtomType.MEMBER_LINK),
        ("membrane-computing", "cognitive-architecture", AtomType.MEMBER_LINK),
        ("echo-state-networks", "cognitive-architecture", AtomType.MEMBER_LINK),
        ("agent-arena-relation", "deep-tree-echo", AtomType.MEMBER_LINK),
        ("attention-allocation", "cognitive-synergy", AtomType.MEMBER_LINK),
        ("pattern-recognition", "cognitive-synergy", AtomType.MEMBER_LINK),
        ("symbolic-reasoning", "cognitive-synergy", AtomType.MEMBER_LINK),
        ("neural-architecture-search", "cognitive-synergy", AtomType.MEMBER_LINK)
    ]
    
    for source_name, target_name, link_type in relationships:
        source_atoms = atomspace.get_atoms_by_name(source_name)
        target_atoms = atomspace.get_atoms_by_name(target_name)
        
        if source_atoms and target_atoms:
            atomspace.add_link(
                link_type,
                [source_atoms[0].id, target_atoms[0].id],
                TruthValue(0.9, 0.85),
                initial_sti=40.0
            )
    
    logger.info("Deep Tree Echo atomspace concepts initialized")


# Example usage and integration demo
async def demo_opencog_deep_tree_echo():
    """Demonstration of OpenCog Deep Tree Echo integration."""
    print("=== OpenCog Deep Tree Echo Integration Demo ===\n")
    
    # Initialize system
    config = OpenCogDeepTreeEchoConfig(
        enable_ecan=True,
        asmoses_population_size=50,
        asmoses_max_generations=10
    )
    
    system = OpenCogDeepTreeEcho(config)
    await system.initialize()
    await system.start()
    
    # Add some concepts
    print("1. Adding concepts...")
    await system.add_concept("neural-network-layer", initial_attention=80.0)
    await system.add_concept("attention-mechanism", initial_attention=90.0)
    await system.add_concept("transformer-architecture", initial_attention=85.0)
    
    # Create relationships
    print("2. Creating relationships...")
    await system.add_relationship(
        "attention-mechanism",
        "transformer-architecture",
        AtomType.MEMBER_LINK
    )
    await system.add_relationship(
        "neural-network-layer",
        "transformer-architecture",
        AtomType.MEMBER_LINK
    )
    
    # Query related concepts
    print("\n3. Finding related concepts...")
    related = await system.find_related_concepts("transformer-architecture", depth=2)
    print(f"Found {len(related)} related concepts")
    
    # Get attentional focus
    print("\n4. Attentional focus:")
    focus_atoms = await system.get_attentional_focus(top_k=5)
    for atom in focus_atoms:
        print(f"  - {atom.name}: STI={atom.attention.sti:.2f}")
    
    # Spread attention
    print("\n5. Spreading attention from 'transformer-architecture'...")
    await system.spread_attention_from("transformer-architecture", depth=2)
    
    # Get updated focus
    print("\n6. Updated attentional focus:")
    focus_atoms = await system.get_attentional_focus(top_k=5)
    for atom in focus_atoms:
        print(f"  - {atom.name}: STI={atom.attention.sti:.2f}")
    
    # Evolve architecture
    print("\n7. Evolving architecture with ASMOSES...")
    result = await system.evolve_architecture(
        task_specs={'task': 'classification', 'input_dim': 512},
        performance_metrics={'accuracy': 0.85}
    )
    print(f"Evolution result: fitness={result['fitness']:.4f}")
    
    # Get statistics
    print("\n8. System statistics:")
    stats = system.get_statistics()
    print(f"  Total atoms: {stats['atomspace']['total_atoms']}")
    print(f"  ECAN STI allocated: {stats['atomspace'].get('ecan', {}).get('sti_allocated', 0):.2f}")
    print(f"  ASMOSES generation: {stats['asmoses']['generation']}")
    
    # Cleanup
    await system.stop()
    print("\n=== Demo completed ===")


if __name__ == "__main__":
    asyncio.run(demo_opencog_deep_tree_echo())
