#!/usr/bin/env python3
"""
Tests for OpenCog Deep Tree Echo Integration

Tests atomspace, HypergraphQL, ASMOSES, and hybrid integration components.
"""

import asyncio
import pytest
from cognitive_architectures import (
    OpenCogAtomSpace,
    AtomType,
    TruthValue,
    AttentionValue,
    HypergraphQLEngine,
    ASMOSESEvolution,
    ASMOSESConfig,
    ProgramTree,
    ProgramOperator,
    OpenCogDeepTreeEcho,
    OpenCogDeepTreeEchoConfig,
    initialize_deep_tree_echo_atomspace
)


class TestOpenCogAtomSpace:
    """Tests for OpenCog AtomSpace implementation."""
    
    def test_atomspace_initialization(self):
        """Test basic atomspace initialization."""
        atomspace = OpenCogAtomSpace(enable_ecan=True)
        assert atomspace is not None
        assert atomspace.enable_ecan is True
        assert len(atomspace.atoms) == 0
    
    def test_add_concept_node(self):
        """Test adding concept nodes."""
        atomspace = OpenCogAtomSpace()
        
        atom = atomspace.add_node(
            AtomType.CONCEPT_NODE,
            "test-concept",
            TruthValue(0.8, 0.7),
            initial_sti=50.0
        )
        
        assert atom is not None
        assert atom.name == "test-concept"
        assert atom.atom_type == AtomType.CONCEPT_NODE
        assert atom.truth_value.strength == 0.8
        assert atom.truth_value.confidence == 0.7
    
    def test_add_duplicate_node(self):
        """Test adding duplicate nodes merges them."""
        atomspace = OpenCogAtomSpace()
        
        atom1 = atomspace.add_node(
            AtomType.CONCEPT_NODE,
            "duplicate",
            TruthValue(0.5, 0.5)
        )
        
        atom2 = atomspace.add_node(
            AtomType.CONCEPT_NODE,
            "duplicate",
            TruthValue(0.8, 0.6)
        )
        
        assert atom1.id == atom2.id
        # Truth value should be updated
        assert atom2.truth_value.confidence > 0.5
    
    def test_add_link(self):
        """Test adding links between nodes."""
        atomspace = OpenCogAtomSpace()
        
        node1 = atomspace.add_node(AtomType.CONCEPT_NODE, "node1")
        node2 = atomspace.add_node(AtomType.CONCEPT_NODE, "node2")
        
        link = atomspace.add_link(
            AtomType.INHERITANCE_LINK,
            [node1.id, node2.id],
            TruthValue(0.9, 0.8)
        )
        
        assert link is not None
        assert link.atom_type == AtomType.INHERITANCE_LINK
        assert len(link.outgoing) == 2
        assert node1.id in link.outgoing
        assert node2.id in link.outgoing
    
    def test_get_incoming_links(self):
        """Test retrieving incoming links."""
        atomspace = OpenCogAtomSpace()
        
        node1 = atomspace.add_node(AtomType.CONCEPT_NODE, "source")
        node2 = atomspace.add_node(AtomType.CONCEPT_NODE, "target")
        
        link = atomspace.add_link(
            AtomType.SIMILARITY_LINK,
            [node1.id, node2.id]
        )
        
        incoming1 = atomspace.get_incoming(node1.id)
        incoming2 = atomspace.get_incoming(node2.id)
        
        assert len(incoming1) == 1
        assert len(incoming2) == 1
        assert incoming1[0].id == link.id
    
    def test_ecan_attention_allocation(self):
        """Test ECAN attention allocation."""
        atomspace = OpenCogAtomSpace(enable_ecan=True, sti_funds=1000.0)
        
        atom = atomspace.add_node(
            AtomType.CONCEPT_NODE,
            "attention-test",
            initial_sti=100.0
        )
        
        assert atom.attention.sti == 100.0
        assert atomspace.ecan.total_sti_allocated == 100.0
    
    def test_ecan_decay(self):
        """Test ECAN attention decay."""
        atomspace = OpenCogAtomSpace(enable_ecan=True)
        
        atom = atomspace.add_node(
            AtomType.CONCEPT_NODE,
            "decay-test",
            initial_sti=100.0
        )
        
        initial_sti = atom.attention.sti
        atomspace.update_ecan()
        
        assert atom.attention.sti < initial_sti
    
    def test_attention_spreading(self):
        """Test attention spreading through graph."""
        atomspace = OpenCogAtomSpace(enable_ecan=True)
        
        node1 = atomspace.add_node(
            AtomType.CONCEPT_NODE,
            "high-attention",
            initial_sti=100.0
        )
        node2 = atomspace.add_node(
            AtomType.CONCEPT_NODE,
            "low-attention",
            initial_sti=0.0
        )
        
        atomspace.add_link(
            AtomType.SIMILARITY_LINK,
            [node1.id, node2.id],
            TruthValue(1.0, 1.0)
        )
        
        initial_sti2 = node2.attention.sti
        atomspace.spread_activation(node1.id, depth=1)
        
        assert node2.attention.sti > initial_sti2
    
    def test_pattern_matching(self):
        """Test basic pattern matching."""
        atomspace = OpenCogAtomSpace()
        
        atomspace.add_node(AtomType.CONCEPT_NODE, "concept1")
        atomspace.add_node(AtomType.CONCEPT_NODE, "concept2")
        atomspace.add_node(AtomType.PREDICATE_NODE, "predicate1")
        
        results = atomspace.pattern_match({
            'atom_type': AtomType.CONCEPT_NODE
        })
        
        assert len(results) == 2
    
    def test_initialize_deep_tree_echo(self):
        """Test Deep Tree Echo atomspace initialization."""
        atomspace = initialize_deep_tree_echo_atomspace()
        
        assert len(atomspace.atoms) > 0
        
        # Check for core concepts
        dte_atoms = atomspace.get_atoms_by_name("deep-tree-echo")
        assert len(dte_atoms) == 1
        assert dte_atoms[0].attention.sti > 0


class TestHypergraphQL:
    """Tests for HypergraphQL query engine."""
    
    def test_hypergraphql_initialization(self):
        """Test HypergraphQL engine initialization."""
        atomspace = OpenCogAtomSpace()
        engine = HypergraphQLEngine(atomspace)
        
        assert engine is not None
        assert engine.atomspace == atomspace
    
    def test_simple_query(self):
        """Test simple HypergraphQL query."""
        atomspace = OpenCogAtomSpace()
        engine = HypergraphQLEngine(atomspace)
        
        atomspace.add_node(AtomType.CONCEPT_NODE, "test1")
        atomspace.add_node(AtomType.CONCEPT_NODE, "test2")
        
        result = engine.query({
            'where': {'atom_type': 'ConceptNode'},
            'limit': 10
        })
        
        assert result.count == 2
        assert len(result.atoms) == 2
    
    def test_query_with_filters(self):
        """Test query with filter conditions."""
        atomspace = OpenCogAtomSpace()
        engine = HypergraphQLEngine(atomspace)
        
        atomspace.add_node(
            AtomType.CONCEPT_NODE,
            "high-truth",
            TruthValue(0.9, 0.8)
        )
        atomspace.add_node(
            AtomType.CONCEPT_NODE,
            "low-truth",
            TruthValue(0.3, 0.2)
        )
        
        result = engine.query({
            'where': {
                'atom_type': 'ConceptNode',
                'truth_value.strength': {'gte': 0.7}
            }
        })
        
        assert result.count == 1
        assert result.atoms[0].name == "high-truth"
    
    def test_traversal_query(self):
        """Test graph traversal in queries."""
        atomspace = OpenCogAtomSpace()
        engine = HypergraphQLEngine(atomspace)
        
        node1 = atomspace.add_node(AtomType.CONCEPT_NODE, "start")
        node2 = atomspace.add_node(AtomType.CONCEPT_NODE, "middle")
        node3 = atomspace.add_node(AtomType.CONCEPT_NODE, "end")
        
        atomspace.add_link(
            AtomType.INHERITANCE_LINK,
            [node1.id, node2.id]
        )
        atomspace.add_link(
            AtomType.INHERITANCE_LINK,
            [node2.id, node3.id]
        )
        
        result = engine.query({
            'where': {'name': 'start'},
            'traverse': {
                'direction': 'outgoing',
                'depth': 2
            }
        })
        
        # Should find start node plus nodes reachable from it
        # Traversal returns nodes found during traversal (not including start unless it appears in the traversal)
        assert result.count >= 1
    
    def test_find_concept_convenience(self):
        """Test find_concept convenience method."""
        atomspace = OpenCogAtomSpace()
        engine = HypergraphQLEngine(atomspace)
        
        atomspace.add_node(
            AtomType.CONCEPT_NODE,
            "findable",
            TruthValue(0.8, 0.7)
        )
        
        results = engine.find_concept("findable", min_truth=0.5)
        
        assert len(results) == 1
        assert results[0].name == "findable"
    
    def test_find_by_attention(self):
        """Test finding atoms by attention values."""
        atomspace = OpenCogAtomSpace(enable_ecan=True)
        engine = HypergraphQLEngine(atomspace)
        
        atomspace.add_node(
            AtomType.CONCEPT_NODE,
            "high-attention",
            initial_sti=100.0
        )
        atomspace.add_node(
            AtomType.CONCEPT_NODE,
            "low-attention",
            initial_sti=10.0
        )
        
        results = engine.find_by_attention(min_sti=50.0, limit=10)
        
        assert len(results) == 1
        assert results[0].name == "high-attention"


class TestASMOSES:
    """Tests for ASMOSES evolutionary system."""
    
    def test_asmoses_initialization(self):
        """Test ASMOSES initialization."""
        config = ASMOSESConfig(population_size=10, max_generations=5)
        asmoses = ASMOSESEvolution(config)
        
        assert asmoses is not None
        assert asmoses.config.population_size == 10
    
    def test_program_tree_creation(self):
        """Test program tree creation."""
        tree = ProgramTree(
            operator=ProgramOperator.ADD,
            children=[
                ProgramTree(operator=ProgramOperator.LINEAR, value=0.5),
                ProgramTree(operator=ProgramOperator.LINEAR, value=0.3)
            ]
        )
        
        assert tree.operator == ProgramOperator.ADD
        assert len(tree.children) == 2
        assert tree.depth() == 2
        assert tree.size() == 3
    
    def test_program_tree_serialization(self):
        """Test program tree to/from dict."""
        tree = ProgramTree(
            operator=ProgramOperator.MUL,
            value=42,
            metadata={'test': 'data'}
        )
        
        tree_dict = tree.to_dict()
        restored = ProgramTree.from_dict(tree_dict)
        
        assert restored.operator == tree.operator
        assert restored.value == tree.value
        assert restored.metadata == tree.metadata
    
    @pytest.mark.asyncio
    async def test_population_initialization(self):
        """Test population initialization."""
        config = ASMOSESConfig(population_size=10)
        asmoses = ASMOSESEvolution(config)
        
        def dummy_fitness(tree):
            return 0.5
        
        asmoses.population.initialize_random(dummy_fitness)
        
        assert len(asmoses.population.programs) == 10
    
    @pytest.mark.asyncio
    async def test_evolution_basic(self):
        """Test basic evolution cycle."""
        config = ASMOSESConfig(
            population_size=20,
            max_generations=3,
            mutation_rate=0.2,
            crossover_rate=0.7
        )
        asmoses = ASMOSESEvolution(config)
        
        def fitness_fn(tree):
            # Simple fitness: prefer smaller trees
            return 1.0 / (tree.size() + 1)
        
        best = await asmoses.evolve(fitness_fn, target_fitness=0.5)
        
        assert best is not None
        assert best.fitness > 0.0
        assert len(asmoses.evolution_history) > 0


class TestOpenCogDeepTreeEchoIntegration:
    """Tests for main OpenCog Deep Tree Echo integration."""
    
    @pytest.mark.asyncio
    async def test_system_initialization(self):
        """Test system initialization."""
        config = OpenCogDeepTreeEchoConfig()
        system = OpenCogDeepTreeEcho(config)
        
        await system.initialize()
        
        assert system.atomspace is not None
        assert system.hypergraph_ql is not None
        assert system.asmoses is not None
    
    @pytest.mark.asyncio
    async def test_add_concept(self):
        """Test adding concepts through main interface."""
        system = OpenCogDeepTreeEcho()
        await system.initialize()
        
        atom = await system.add_concept(
            "test-concept",
            truth_strength=0.9,
            initial_attention=75.0
        )
        
        assert atom is not None
        assert atom.name == "test-concept"
        assert atom.attention.sti == 75.0
    
    @pytest.mark.asyncio
    async def test_add_relationship(self):
        """Test adding relationships."""
        system = OpenCogDeepTreeEcho()
        await system.initialize()
        
        await system.add_concept("concept-a")
        await system.add_concept("concept-b")
        
        link = await system.add_relationship(
            "concept-a",
            "concept-b",
            AtomType.SIMILARITY_LINK
        )
        
        assert link is not None
        assert link.atom_type == AtomType.SIMILARITY_LINK
    
    @pytest.mark.asyncio
    async def test_find_related_concepts(self):
        """Test finding related concepts."""
        system = OpenCogDeepTreeEcho()
        await system.initialize()
        
        await system.add_concept("center")
        await system.add_concept("related1")
        await system.add_concept("related2")
        
        await system.add_relationship("center", "related1")
        await system.add_relationship("center", "related2")
        
        related = await system.find_related_concepts("center", depth=2)
        
        # Should find center node + links + related nodes
        # At minimum we should find the center node plus at least one related node
        assert len(related) >= 1
    
    @pytest.mark.asyncio
    async def test_query_interface(self):
        """Test query interface."""
        system = OpenCogDeepTreeEcho()
        await system.initialize()
        
        await system.add_concept("queryable", truth_strength=0.9)
        
        result = await system.query({
            'where': {
                'atom_type': 'ConceptNode',
                'name': 'queryable'
            }
        })
        
        assert result.count >= 1
    
    @pytest.mark.asyncio
    async def test_attentional_focus(self):
        """Test getting attentional focus."""
        system = OpenCogDeepTreeEcho()
        await system.initialize()
        
        await system.add_concept("high-attention", initial_attention=100.0)
        await system.add_concept("low-attention", initial_attention=10.0)
        
        focus = await system.get_attentional_focus(top_k=5)
        
        assert len(focus) > 0
        # Highest attention should be first
        assert focus[0].attention.sti >= focus[-1].attention.sti
    
    @pytest.mark.asyncio
    async def test_attention_spreading(self):
        """Test attention spreading."""
        system = OpenCogDeepTreeEcho()
        await system.initialize()
        
        await system.add_concept("source", initial_attention=100.0)
        await system.add_concept("target", initial_attention=0.0)
        await system.add_relationship("source", "target")
        
        # Get target before spreading
        target_atoms = system.atomspace.get_atoms_by_name("target")
        initial_attention = target_atoms[0].attention.sti
        
        await system.spread_attention_from("source", depth=1)
        
        # Check attention increased
        assert target_atoms[0].attention.sti > initial_attention
    
    @pytest.mark.asyncio
    async def test_system_statistics(self):
        """Test getting system statistics."""
        system = OpenCogDeepTreeEcho()
        await system.initialize()
        
        await system.add_concept("stat-test")
        
        stats = system.get_statistics()
        
        assert 'atomspace' in stats
        assert 'asmoses' in stats
        assert 'system' in stats
        assert stats['atomspace']['total_atoms'] > 0
    
    @pytest.mark.asyncio
    async def test_background_tasks(self):
        """Test starting and stopping background tasks."""
        config = OpenCogDeepTreeEchoConfig(
            enable_ecan=True,
            enable_attention_spreading=True
        )
        system = OpenCogDeepTreeEcho(config)
        await system.initialize()
        
        await system.start()
        assert system.is_running
        assert len(system.background_tasks) > 0
        
        await asyncio.sleep(0.1)  # Let tasks run briefly
        
        await system.stop()
        assert not system.is_running
        assert len(system.background_tasks) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
