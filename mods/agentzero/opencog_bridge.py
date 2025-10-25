#!/usr/bin/env python3
"""
OpenCog Bridge for Agent-Zero Integration

Provides integration between agent-zero's multi-agent framework and OpenCog's
cognitive architecture, including HypergraphQL ECAN-aware atomspace.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class HypergraphQLAgentInterface:
    """
    HypergraphQL interface for agent queries and knowledge access.
    
    Provides agents with query capabilities over the OpenCog atomspace
    using HypergraphQL pattern matching.
    """
    
    def __init__(self, atomspace):
        self.atomspace = atomspace
        self.query_cache: Dict[str, Any] = {}
        
    async def query(self, query_string: str) -> Dict[str, Any]:
        """
        Execute a HypergraphQL query on the atomspace.
        
        Args:
            query_string: HypergraphQL query
            
        Returns:
            Query results
        """
        # Check cache
        if query_string in self.query_cache:
            logger.debug(f"Returning cached query result for: {query_string[:50]}")
            return self.query_cache[query_string]
            
        # Execute query through hypergraph_ql engine
        try:
            from cognitive_architectures.hypergraph_ql import HypergraphQLEngine
            
            engine = HypergraphQLEngine(self.atomspace)
            result = await asyncio.to_thread(engine.execute, query_string)
            
            # Cache result
            self.query_cache[query_string] = result
            
            return result
            
        except Exception as e:
            logger.error(f"HypergraphQL query error: {e}")
            return {
                'success': False,
                'error': str(e),
                'results': []
            }
            
    async def store_agent_memory(
        self,
        agent_id: str,
        memory_type: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Store agent memory in atomspace with ECAN attention allocation.
        
        Args:
            agent_id: Agent identifier
            memory_type: Type of memory (episodic, semantic, procedural)
            content: Memory content
            metadata: Optional metadata
        """
        try:
            from cognitive_architectures.opencog_atomspace import (
                AtomType, TruthValue
            )
            
            # Create concept node for agent
            agent_atom = self.atomspace.add_node(
                AtomType.CONCEPT_NODE,
                f"agent-{agent_id}",
                TruthValue(strength=0.95, confidence=0.9),
                initial_sti=100.0
            )
            
            # Create memory node
            memory_atom = self.atomspace.add_node(
                AtomType.CONCEPT_NODE,
                f"memory-{memory_type}-{content[:50]}",
                TruthValue(strength=0.9, confidence=0.85),
                initial_sti=80.0
            )
            
            # Link agent to memory
            link = self.atomspace.add_link(
                AtomType.INHERITANCE_LINK,
                [agent_atom.id, memory_atom.id],
                TruthValue(strength=0.95, confidence=0.9)
            )
            
            logger.info(f"Stored {memory_type} memory for agent {agent_id}")
            
        except Exception as e:
            logger.error(f"Error storing agent memory: {e}")
            
    async def retrieve_agent_memories(
        self,
        agent_id: str,
        memory_type: Optional[str] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Retrieve agent memories from atomspace.
        
        Args:
            agent_id: Agent identifier
            memory_type: Optional memory type filter
            limit: Maximum number of memories to retrieve
            
        Returns:
            List of memory entries
        """
        query = f"""
        {{
            concepts(filter: {{name: "agent-{agent_id}"}}) {{
                id
                name
                outgoing {{
                    target {{
                        id
                        name
                        truthValue {{
                            strength
                            confidence
                        }}
                    }}
                }}
            }}
        }}
        """
        
        result = await self.query(query)
        
        # Extract and format memories
        memories = []
        # Parse result and extract relevant memories
        # This would be implemented based on actual HypergraphQL result structure
        
        return memories[:limit]


class OpenCogAgentBridge:
    """
    Bridge between agent-zero agents and OpenCog cognitive architecture.
    
    Provides:
    - ECAN-aware attention allocation for agent knowledge
    - HypergraphQL query interface for agents
    - Knowledge persistence in atomspace
    - ASMOSES integration for agent evolution
    """
    
    def __init__(self, config):
        self.config = config
        self.atomspace = None
        self.hypergraph_ql: Optional[HypergraphQLAgentInterface] = None
        self.asmoses_bridge = None
        self.agent_atoms: Dict[str, Any] = {}
        self.initialized = False
        
    async def initialize(self):
        """Initialize OpenCog components."""
        if self.initialized:
            return
            
        logger.info("Initializing OpenCog Agent Bridge")
        
        try:
            from cognitive_architectures.opencog_atomspace import (
                OpenCogAtomSpace,
                initialize_deep_tree_echo_atomspace
            )
            from cognitive_architectures.hypergraph_ql import HypergraphQLEngine
            from cognitive_architectures.asmoses_bridge import (
                ASMOSESEvolution,
                ASMOSESConfig,
                HybridASMOSESAphroditeIntegration
            )
            
            # Initialize atomspace with ECAN
            self.atomspace = OpenCogAtomSpace(
                enable_ecan=self.config.enable_ecan,
                sti_funds=100000.0,
                lti_funds=100000.0
            )
            
            # Initialize Deep Tree Echo atomspace structures
            initialize_deep_tree_echo_atomspace(self.atomspace)
            
            # Initialize HypergraphQL interface
            self.hypergraph_ql = HypergraphQLAgentInterface(self.atomspace)
            
            # Initialize ASMOSES if enabled
            if self.config.enable_asmoses:
                asmoses_config = ASMOSESConfig(
                    population_size=50,
                    max_generations=30,
                    mutation_rate=0.1,
                    enable_atomspace_storage=True
                )
                self.asmoses_bridge = HybridASMOSESAphroditeIntegration(
                    asmoses_config,
                    self.atomspace
                )
                
            logger.info("OpenCog Agent Bridge initialized successfully")
            self.initialized = True
            
        except ImportError as e:
            logger.warning(f"Could not initialize OpenCog components: {e}")
            logger.info("Running without OpenCog integration")
            
    async def register_agent(self, agent_id: str, agent: Any):
        """
        Register an agent in the OpenCog atomspace.
        
        Args:
            agent_id: Agent identifier
            agent: Agent instance
        """
        if not self.initialized or not self.atomspace:
            logger.warning("OpenCog not initialized, skipping agent registration")
            return
            
        try:
            from cognitive_architectures.opencog_atomspace import (
                AtomType, TruthValue
            )
            
            # Create agent atom in atomspace
            agent_atom = self.atomspace.add_node(
                AtomType.CONCEPT_NODE,
                f"agent-zero-{agent_id}",
                TruthValue(strength=0.95, confidence=0.9),
                initial_sti=150.0  # High initial attention for active agents
            )
            
            # Add agent capabilities as linked concepts
            capabilities = [
                "multi-agent-communication",
                "code-execution",
                "knowledge-access",
                "task-decomposition"
            ]
            
            for capability in capabilities:
                capability_atom = self.atomspace.add_node(
                    AtomType.CONCEPT_NODE,
                    capability,
                    TruthValue(strength=0.9, confidence=0.85)
                )
                
                self.atomspace.add_link(
                    AtomType.INHERITANCE_LINK,
                    [agent_atom.id, capability_atom.id],
                    TruthValue(strength=0.9, confidence=0.9)
                )
                
            self.agent_atoms[agent_id] = agent_atom
            logger.info(f"Registered agent {agent_id} in OpenCog atomspace")
            
        except Exception as e:
            logger.error(f"Error registering agent in OpenCog: {e}")
            
    async def enhance_agent_reasoning(
        self,
        agent_id: str,
        query: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Enhance agent reasoning using OpenCog PLN (Probabilistic Logic Networks).
        
        Args:
            agent_id: Agent making the query
            query: Reasoning query
            context: Optional context
            
        Returns:
            Enhanced reasoning results
        """
        if not self.hypergraph_ql:
            return {'enhanced': False, 'result': None}
            
        # Query atomspace for relevant knowledge
        knowledge = await self.hypergraph_ql.retrieve_agent_memories(agent_id)
        
        # Perform attention spreading from agent node
        if agent_id in self.agent_atoms:
            self.atomspace.spread_activation(
                self.agent_atoms[agent_id].id,
                depth=3,
                spread_strength=0.7
            )
            
        # Get atoms in attentional focus
        focus_atoms = self.atomspace.get_attentional_focus(top_k=20)
        
        result = {
            'enhanced': True,
            'knowledge_retrieved': len(knowledge),
            'focus_atoms': len(focus_atoms),
            'context': context or {},
            'reasoning': {
                'query': query,
                'agent_id': agent_id
            }
        }
        
        return result
        
    async def evolve_agent_architecture(
        self,
        agent_id: str,
        performance_metrics: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Use ASMOSES to evolve agent architecture based on performance.
        
        Args:
            agent_id: Agent to evolve
            performance_metrics: Performance data
            
        Returns:
            Evolution results
        """
        if not self.asmoses_bridge:
            return {'evolved': False, 'reason': 'ASMOSES not enabled'}
            
        logger.info(f"Evolving architecture for agent {agent_id}")
        
        # This would integrate with ASMOSES evolution
        # For now, return placeholder result
        result = {
            'evolved': True,
            'agent_id': agent_id,
            'performance_metrics': performance_metrics,
            'evolution_generation': 0
        }
        
        return result
        
    async def shutdown(self):
        """Shutdown OpenCog bridge."""
        logger.info("Shutting down OpenCog Agent Bridge")
        
        if self.atomspace:
            # Persist atomspace if needed
            pass
            
        self.initialized = False
