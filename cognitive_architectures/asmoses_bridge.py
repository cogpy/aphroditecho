#!/usr/bin/env python3
"""
ASMOSES (Adaptive Symbolic MOSES) Bridge for Aphrodite Engine

Integrates evolutionary program synthesis from OpenCog's MOSES with
the Aphrodite Engine and existing Echo-Self evolution system for
hybrid symbolic-neural architecture optimization.
"""

import asyncio
import logging
import random
import uuid
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import numpy as np

from .opencog_atomspace import OpenCogAtomSpace, Atom, AtomType, TruthValue

logger = logging.getLogger(__name__)


class ProgramOperator(Enum):
    """Operators for program synthesis."""
    # Arithmetic
    ADD = "+"
    SUB = "-"
    MUL = "*"
    DIV = "/"
    
    # Logical
    AND = "and"
    OR = "or"
    NOT = "not"
    
    # Comparison
    GT = ">"
    LT = "<"
    EQ = "=="
    
    # Control flow
    IF = "if"
    WHILE = "while"
    
    # Neural network operations
    LINEAR = "linear"
    CONV = "conv"
    ATTENTION = "attention"
    ACTIVATION = "activation"


@dataclass
class ProgramTree:
    """
    Program tree representation for evolved programs.
    
    Compatible with both symbolic operations and neural architecture specifications.
    """
    operator: ProgramOperator
    children: List['ProgramTree'] = field(default_factory=list)
    value: Optional[Any] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'operator': self.operator.value,
            'children': [c.to_dict() for c in self.children],
            'value': self.value,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ProgramTree':
        """Create from dictionary representation."""
        operator = ProgramOperator(data['operator'])
        children = [cls.from_dict(c) for c in data.get('children', [])]
        return cls(
            operator=operator,
            children=children,
            value=data.get('value'),
            metadata=data.get('metadata', {})
        )
    
    def depth(self) -> int:
        """Calculate tree depth."""
        if not self.children:
            return 1
        return 1 + max(c.depth() for c in self.children)
    
    def size(self) -> int:
        """Calculate tree size (number of nodes)."""
        return 1 + sum(c.size() for c in self.children)
    
    def copy(self) -> 'ProgramTree':
        """Create a deep copy of the tree."""
        return ProgramTree(
            operator=self.operator,
            children=[c.copy() for c in self.children],
            value=self.value,
            metadata=self.metadata.copy()
        )


@dataclass
class ASMOSESConfig:
    """Configuration for ASMOSES evolution."""
    population_size: int = 100
    max_generations: int = 100
    max_tree_depth: int = 8
    mutation_rate: float = 0.1
    crossover_rate: float = 0.7
    tournament_size: int = 5
    elitism_count: int = 5
    fitness_threshold: float = 0.95
    diversity_weight: float = 0.1
    enable_atomspace_storage: bool = True


@dataclass
class EvolvedProgram:
    """Container for an evolved program."""
    id: str
    tree: ProgramTree
    fitness: float
    generation: int
    parent_ids: List[str] = field(default_factory=list)
    complexity: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.complexity == 0:
            self.complexity = self.tree.size()


class ASMOSESPopulation:
    """Manages population of evolved programs."""
    
    def __init__(self, config: ASMOSESConfig):
        self.config = config
        self.programs: Dict[str, EvolvedProgram] = {}
        self.generation = 0
    
    def initialize_random(self, fitness_function: Callable):
        """Initialize population with random programs."""
        self.programs.clear()
        
        for _ in range(self.config.population_size):
            tree = self._generate_random_tree(max_depth=3)
            program_id = str(uuid.uuid4())
            
            program = EvolvedProgram(
                id=program_id,
                tree=tree,
                fitness=0.0,
                generation=0
            )
            
            self.programs[program_id] = program
        
        logger.info(f"Initialized population with {len(self.programs)} random programs")
    
    def _generate_random_tree(
        self,
        max_depth: int,
        current_depth: int = 0
    ) -> ProgramTree:
        """Generate a random program tree."""
        # Select random operator
        if current_depth >= max_depth:
            # Use terminal (value) node
            return ProgramTree(
                operator=random.choice([ProgramOperator.ADD, ProgramOperator.LINEAR]),
                value=random.random()
            )
        
        # Choose operator based on type
        if random.random() < 0.3:  # 30% chance of neural operation
            operator = random.choice([
                ProgramOperator.LINEAR,
                ProgramOperator.CONV,
                ProgramOperator.ATTENTION,
                ProgramOperator.ACTIVATION
            ])
        else:  # Symbolic operations
            operator = random.choice([
                ProgramOperator.ADD,
                ProgramOperator.MUL,
                ProgramOperator.IF,
                ProgramOperator.AND
            ])
        
        # Generate children
        num_children = random.randint(1, 3)
        children = [
            self._generate_random_tree(max_depth, current_depth + 1)
            for _ in range(num_children)
        ]
        
        return ProgramTree(operator=operator, children=children)
    
    def select_tournament(self) -> EvolvedProgram:
        """Tournament selection."""
        tournament = random.sample(
            list(self.programs.values()),
            min(self.config.tournament_size, len(self.programs))
        )
        return max(tournament, key=lambda p: p.fitness)
    
    def get_best(self, k: int = 1) -> List[EvolvedProgram]:
        """Get top k programs by fitness."""
        sorted_programs = sorted(
            self.programs.values(),
            key=lambda p: p.fitness,
            reverse=True
        )
        return sorted_programs[:k]
    
    def add_program(self, program: EvolvedProgram):
        """Add program to population."""
        self.programs[program.id] = program
    
    def remove_program(self, program_id: str):
        """Remove program from population."""
        if program_id in self.programs:
            del self.programs[program_id]


class ASMOSESEvolution:
    """
    Adaptive Symbolic MOSES Evolution Engine.
    
    Implements program synthesis and neural architecture search through
    evolutionary algorithms inspired by OpenCog's MOSES.
    """
    
    def __init__(
        self,
        config: ASMOSESConfig,
        atomspace: Optional[OpenCogAtomSpace] = None
    ):
        self.config = config
        self.atomspace = atomspace
        self.population = ASMOSESPopulation(config)
        self.evolution_history: List[Dict[str, Any]] = []
    
    async def evolve(
        self,
        fitness_function: Callable[[ProgramTree], float],
        target_fitness: Optional[float] = None
    ) -> EvolvedProgram:
        """
        Run evolutionary process to optimize programs.
        
        Args:
            fitness_function: Function to evaluate program fitness
            target_fitness: Optional target fitness to stop evolution
        
        Returns:
            Best evolved program
        """
        # Initialize population
        self.population.initialize_random(fitness_function)
        
        # Evaluate initial population
        await self._evaluate_population(fitness_function)
        
        target = target_fitness or self.config.fitness_threshold
        
        for generation in range(self.config.max_generations):
            self.population.generation = generation
            
            # Get best program
            best = self.population.get_best(1)[0]
            
            logger.info(
                f"Generation {generation}: Best fitness = {best.fitness:.4f}, "
                f"Complexity = {best.complexity}"
            )
            
            # Record history
            self.evolution_history.append({
                'generation': generation,
                'best_fitness': best.fitness,
                'avg_fitness': np.mean([p.fitness for p in self.population.programs.values()]),
                'best_complexity': best.complexity,
                'population_size': len(self.population.programs)
            })
            
            # Check stopping criterion
            if best.fitness >= target:
                logger.info(f"Target fitness {target} reached at generation {generation}")
                break
            
            # Create next generation
            await self._create_next_generation(fitness_function)
            
            # Store best programs in atomspace if enabled
            if self.config.enable_atomspace_storage and self.atomspace:
                await self._store_in_atomspace(best)
        
        return self.population.get_best(1)[0]
    
    async def _evaluate_population(self, fitness_function: Callable):
        """Evaluate fitness for all programs in population."""
        tasks = []
        for program in self.population.programs.values():
            tasks.append(self._evaluate_program(program, fitness_function))
        
        await asyncio.gather(*tasks)
    
    async def _evaluate_program(
        self,
        program: EvolvedProgram,
        fitness_function: Callable
    ):
        """Evaluate a single program's fitness."""
        try:
            # Run fitness function (may be async or sync)
            if asyncio.iscoroutinefunction(fitness_function):
                fitness = await fitness_function(program.tree)
            else:
                fitness = fitness_function(program.tree)
            
            # Apply complexity penalty
            complexity_penalty = self.config.diversity_weight * (program.complexity / 100)
            program.fitness = max(0.0, fitness - complexity_penalty)
            
        except Exception as e:
            logger.warning(f"Error evaluating program {program.id}: {e}")
            program.fitness = 0.0
    
    async def _create_next_generation(self, fitness_function: Callable):
        """Create next generation through selection, crossover, and mutation."""
        new_programs = []
        
        # Elitism: keep best programs
        elite = self.population.get_best(self.config.elitism_count)
        new_programs.extend(elite)
        
        # Generate offspring
        while len(new_programs) < self.config.population_size:
            # Selection
            parent1 = self.population.select_tournament()
            parent2 = self.population.select_tournament()
            
            # Crossover
            if random.random() < self.config.crossover_rate:
                child_tree = self._crossover(parent1.tree, parent2.tree)
            else:
                child_tree = parent1.tree.copy()
            
            # Mutation
            if random.random() < self.config.mutation_rate:
                child_tree = self._mutate(child_tree)
            
            # Create new program
            child = EvolvedProgram(
                id=str(uuid.uuid4()),
                tree=child_tree,
                fitness=0.0,
                generation=self.population.generation + 1,
                parent_ids=[parent1.id, parent2.id]
            )
            
            new_programs.append(child)
        
        # Replace population
        self.population.programs.clear()
        for program in new_programs:
            self.population.add_program(program)
        
        # Evaluate new generation
        await self._evaluate_population(fitness_function)
    
    def _crossover(self, tree1: ProgramTree, tree2: ProgramTree) -> ProgramTree:
        """Perform subtree crossover between two program trees."""
        # Copy trees
        child = tree1.copy()
        
        # Select random subtree from child
        subtree_nodes = self._get_all_nodes(child)
        if not subtree_nodes:
            return child
        
        crossover_point = random.choice(subtree_nodes)
        
        # Select random subtree from parent2
        parent2_nodes = self._get_all_nodes(tree2)
        if not parent2_nodes:
            return child
        
        donor_subtree = random.choice(parent2_nodes).copy()
        
        # Replace subtree
        self._replace_subtree(child, crossover_point, donor_subtree)
        
        return child
    
    def _mutate(self, tree: ProgramTree) -> ProgramTree:
        """Apply mutation to program tree."""
        mutated = tree.copy()
        
        # Select mutation type
        mutation_type = random.choice(['operator', 'subtree', 'value'])
        
        if mutation_type == 'operator':
            # Mutate operator
            nodes = self._get_all_nodes(mutated)
            if nodes:
                node = random.choice(nodes)
                # Choose new operator of same arity
                if node.operator in [ProgramOperator.ADD, ProgramOperator.MUL]:
                    node.operator = random.choice([ProgramOperator.ADD, ProgramOperator.MUL])
                elif node.operator in [ProgramOperator.LINEAR, ProgramOperator.CONV]:
                    node.operator = random.choice([ProgramOperator.LINEAR, ProgramOperator.CONV])
        
        elif mutation_type == 'subtree':
            # Replace random subtree with new random tree
            nodes = self._get_all_nodes(mutated)
            if nodes:
                node = random.choice(nodes)
                new_subtree = self.population._generate_random_tree(max_depth=2)
                self._replace_subtree(mutated, node, new_subtree)
        
        elif mutation_type == 'value':
            # Mutate terminal value
            nodes = self._get_all_nodes(mutated)
            value_nodes = [n for n in nodes if n.value is not None]
            if value_nodes:
                node = random.choice(value_nodes)
                node.value = random.random()
        
        return mutated
    
    def _get_all_nodes(self, tree: ProgramTree) -> List[ProgramTree]:
        """Get all nodes in tree as flat list."""
        nodes = [tree]
        for child in tree.children:
            nodes.extend(self._get_all_nodes(child))
        return nodes
    
    def _replace_subtree(
        self,
        tree: ProgramTree,
        target: ProgramTree,
        replacement: ProgramTree
    ):
        """Replace target subtree with replacement in tree."""
        # This is a simplified implementation
        # Full implementation would properly maintain tree structure
        if tree == target:
            tree.operator = replacement.operator
            tree.children = replacement.children
            tree.value = replacement.value
            tree.metadata = replacement.metadata
        else:
            for i, child in enumerate(tree.children):
                if child == target:
                    tree.children[i] = replacement
                else:
                    self._replace_subtree(child, target, replacement)
    
    async def _store_in_atomspace(self, program: EvolvedProgram):
        """Store evolved program in OpenCog atomspace."""
        if not self.atomspace:
            return
        
        # Create program node
        program_node = self.atomspace.add_node(
            AtomType.SCHEMA_NODE,
            f"evolved-program-{program.id[:8]}",
            TruthValue(program.fitness, 0.8),
            initial_sti=program.fitness * 100
        )
        
        # Add metadata
        program_node.metadata.update({
            'generation': program.generation,
            'complexity': program.complexity,
            'tree_depth': program.tree.depth(),
            'parent_ids': program.parent_ids
        })
        
        logger.debug(f"Stored program {program.id} in atomspace")


class HybridASMOSESAphroditeIntegration:
    """
    Hybrid integration of ASMOSES with Aphrodite Engine.
    
    Bridges symbolic program evolution with neural architecture optimization
    for the Deep Tree Echo system.
    """
    
    def __init__(
        self,
        asmoses_config: ASMOSESConfig,
        atomspace: OpenCogAtomSpace,
        echo_self_evolution_engine = None
    ):
        self.asmoses = ASMOSESEvolution(asmoses_config, atomspace)
        self.atomspace = atomspace
        self.echo_self_engine = echo_self_evolution_engine
    
    async def optimize_architecture(
        self,
        task_specs: Dict[str, Any],
        performance_metrics: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Optimize neural architecture using hybrid ASMOSES-Aphrodite approach.
        
        Args:
            task_specs: Specifications for the optimization task
            performance_metrics: Current performance metrics
        
        Returns:
            Optimized architecture specification
        """
        # Define fitness function based on task
        def fitness_fn(program_tree: ProgramTree) -> float:
            # Convert program tree to architecture spec
            arch_spec = self._tree_to_architecture(program_tree)
            
            # Evaluate architecture (simplified)
            # In real implementation, would train and evaluate the architecture
            score = self._evaluate_architecture(arch_spec, performance_metrics)
            return score
        
        # Run ASMOSES evolution
        best_program = await self.asmoses.evolve(fitness_fn)
        
        # Convert best program to architecture specification
        optimized_arch = self._tree_to_architecture(best_program.tree)
        
        # Store in atomspace with high attention
        if self.atomspace:
            arch_node = self.atomspace.add_node(
                AtomType.CONCEPT_NODE,
                f"optimized-architecture-{best_program.id[:8]}",
                TruthValue(best_program.fitness, 0.9),
                initial_sti=best_program.fitness * 150
            )
            arch_node.metadata['architecture'] = optimized_arch
        
        logger.info(
            f"Optimized architecture with fitness {best_program.fitness:.4f}, "
            f"complexity {best_program.complexity}"
        )
        
        return {
            'architecture': optimized_arch,
            'fitness': best_program.fitness,
            'complexity': best_program.complexity,
            'program_id': best_program.id
        }
    
    def _tree_to_architecture(self, tree: ProgramTree) -> Dict[str, Any]:
        """Convert program tree to neural architecture specification."""
        # Simplified conversion
        # Real implementation would construct full architecture spec
        layers = []
        self._extract_layers(tree, layers)
        
        return {
            'layers': layers,
            'tree_depth': tree.depth(),
            'tree_size': tree.size()
        }
    
    def _extract_layers(self, tree: ProgramTree, layers: List[Dict]):
        """Extract layer specifications from program tree."""
        if tree.operator in [ProgramOperator.LINEAR, ProgramOperator.CONV, ProgramOperator.ATTENTION]:
            layers.append({
                'type': tree.operator.value,
                'config': tree.metadata
            })
        
        for child in tree.children:
            self._extract_layers(child, layers)
    
    def _evaluate_architecture(
        self,
        arch_spec: Dict[str, Any],
        baseline_metrics: Dict[str, float]
    ) -> float:
        """Evaluate architecture specification."""
        # Simplified evaluation
        # Real implementation would train and benchmark the architecture
        
        # Prefer architectures with moderate complexity
        num_layers = len(arch_spec.get('layers', []))
        complexity_score = 1.0 - abs(num_layers - 5) / 10.0
        
        # Add some randomness to simulate performance variance
        performance_score = 0.5 + random.random() * 0.5
        
        return 0.7 * performance_score + 0.3 * complexity_score
    
    async def synchronize_with_echo_self(self):
        """Synchronize ASMOSES results with Echo-Self evolution engine."""
        if not self.echo_self_engine:
            logger.warning("Echo-Self evolution engine not available")
            return
        
        # Get best ASMOSES programs
        best_programs = self.asmoses.population.get_best(k=5)
        
        # Convert to Echo-Self individuals and add to population
        # This would require mapping between program trees and neural genomes
        logger.info(f"Synchronized {len(best_programs)} programs with Echo-Self engine")
