#!/usr/bin/env python3
"""
Example usage of Aphroditecho-Zero integration.

This script demonstrates the basic usage of the aphroditecho-zero
agent-zero integration with Aphrodite Engine.
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from mods.agentzero import (
    AphroditechoZero,
    AphroditechoZeroConfig
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def basic_example():
    """Basic example of creating and using an agent."""
    logger.info("=== Basic Aphroditecho-Zero Example ===")
    
    # Create configuration with all features enabled
    config = AphroditechoZeroConfig(
        enable_opencog=True,
        enable_dtesn=True,
        enable_aar=True,
        enable_multi_agent=True,
        max_concurrent_agents=5
    )
    
    # Initialize Aphroditecho-Zero
    logger.info("Initializing Aphroditecho-Zero...")
    az = AphroditechoZero(config)
    await az.initialize()
    
    # Create a single agent
    logger.info("Creating agent...")
    agent = await az.create_agent(
        agent_id="demo-agent-001",
        name="DemoAgent",
        agent_type="general"
    )
    
    # Process an inference request
    logger.info("Processing inference request...")
    result = await az.process_inference_request(
        agent_id="demo-agent-001",
        prompt="Explain the integration of membrane computing with neural networks",
        context={'domain': 'cognitive_architecture', 'detail_level': 'high'}
    )
    
    logger.info(f"Response: {result['response'][:200]}...")
    logger.info(f"Metadata: {result['metadata']}")
    
    # Cleanup
    logger.info("Shutting down...")
    await az.shutdown()
    logger.info("=== Example Complete ===")


async def multi_agent_example():
    """Example of multi-agent coordination."""
    logger.info("=== Multi-Agent Coordination Example ===")
    
    config = AphroditechoZeroConfig(
        enable_aar=True,
        enable_arena_simulation=True,
        enable_relation_graphs=True,
        max_concurrent_agents=10
    )
    
    az = AphroditechoZero(config)
    await az.initialize()
    
    # Create root coordinator agent
    logger.info("Creating root agent...")
    root_agent = await az.create_agent(
        agent_id="coordinator",
        name="CoordinatorAgent"
    )
    
    # Create subordinate worker agents
    logger.info("Creating worker agents...")
    worker_ids = []
    for i in range(3):
        worker_id = f"worker-{i:02d}"
        worker = await az.create_agent(
            agent_id=worker_id,
            name=f"Worker{i}",
            config={'superior': 'coordinator'}
        )
        worker_ids.append(worker_id)
        logger.info(f"Created worker: {worker_id}")
    
    # Coordinate a multi-agent task
    if az.aar_orchestrator:
        logger.info("Coordinating multi-agent task...")
        result = await az.aar_orchestrator.coordinate_multi_agent_task(
            root_agent_id="coordinator",
            task_description="Implement a distributed deep learning training system",
            max_agents=5
        )
        
        logger.info(f"Coordination result: {result['success']}")
        logger.info(f"Network stats: {result['network_stats']}")
        
        # Get agent hierarchy
        logger.info("Agent hierarchy:")
        logger.info(f"{result['hierarchy']}")
    
    await az.shutdown()
    logger.info("=== Multi-Agent Example Complete ===")


async def opencog_integration_example():
    """Example of OpenCog integration features."""
    logger.info("=== OpenCog Integration Example ===")
    
    config = AphroditechoZeroConfig(
        enable_opencog=True,
        enable_ecan=True,
        enable_hypergraph_ql=True,
        enable_asmoses=True
    )
    
    az = AphroditechoZero(config)
    await az.initialize()
    
    # Create agent with OpenCog capabilities
    logger.info("Creating OpenCog-enhanced agent...")
    agent = await az.create_agent(
        agent_id="opencog-agent",
        name="CognitiveAgent"
    )
    
    if az.opencog_bridge:
        # Store some knowledge in atomspace
        logger.info("Storing knowledge in atomspace...")
        if az.opencog_bridge.hypergraph_ql:
            await az.opencog_bridge.hypergraph_ql.store_agent_memory(
                agent_id="opencog-agent",
                memory_type="semantic",
                content="Aphroditecho-Zero integrates agent-zero with OpenCog atomspace",
                metadata={'importance': 0.95, 'source': 'demo'}
            )
            
            await az.opencog_bridge.hypergraph_ql.store_agent_memory(
                agent_id="opencog-agent",
                memory_type="procedural",
                content="Multi-agent coordination uses AAR orchestration",
                metadata={'importance': 0.90, 'source': 'demo'}
            )
        
        # Enhance reasoning with OpenCog
        logger.info("Enhancing reasoning with OpenCog...")
        reasoning_result = await az.opencog_bridge.enhance_agent_reasoning(
            agent_id="opencog-agent",
            query="How does ECAN attention allocation work in multi-agent systems?",
            context={'domain': 'cognitive_architecture'}
        )
        
        logger.info(f"Reasoning enhanced: {reasoning_result['enhanced']}")
        logger.info(f"Focus atoms: {reasoning_result['focus_atoms']}")
        
        # Attempt agent architecture evolution
        logger.info("Evolving agent architecture...")
        evolution_result = await az.opencog_bridge.evolve_agent_architecture(
            agent_id="opencog-agent",
            performance_metrics={
                'accuracy': 0.87,
                'speed': 0.92,
                'memory_efficiency': 0.85
            }
        )
        
        logger.info(f"Evolution result: {evolution_result}")
    
    await az.shutdown()
    logger.info("=== OpenCog Example Complete ===")


async def dtesn_example():
    """Example of Deep Tree Echo identity integration."""
    logger.info("=== Deep Tree Echo Identity Example ===")
    
    config = AphroditechoZeroConfig(
        enable_dtesn=True,
        enable_membrane_computing=True,
        enable_echo_self=True
    )
    
    az = AphroditechoZero(config)
    await az.initialize()
    
    # Create agent with DTESN capabilities
    logger.info("Creating DTESN-enabled agent...")
    agent = await az.create_agent(
        agent_id="dtesn-agent",
        name="EchoAgent"
    )
    
    if az.dtesn_identity:
        # Check if agent is wrapped
        if "dtesn-agent" in az.dtesn_identity.wrapped_agents:
            wrapper = az.dtesn_identity.wrapped_agents["dtesn-agent"]
            logger.info(f"Agent wrapped with DTESN identity")
            
            # Process some temporal data through DTESN
            logger.info("Processing data through DTESN reservoir...")
            test_data = "Temporal sequence: t1, t2, t3, t4, t5"
            result = await wrapper.process_with_dtesn(
                input_data=test_data,
                context={'temporal': True, 'sequence_length': 5}
            )
            
            logger.info(f"DTESN enhanced: {result['dtesn_enhanced']}")
            if 'reservoir_output' in result:
                logger.info(f"Reservoir output shape: {len(result['reservoir_output'])}")
            
            # Evolve agent identity based on performance
            logger.info("Evolving agent identity...")
            evolution = await az.dtesn_identity.evolve_agent_identity(
                agent_id="dtesn-agent",
                performance_data={
                    'prediction_accuracy': 0.89,
                    'temporal_coherence': 0.92,
                    'resource_efficiency': 0.88
                }
            )
            
            logger.info(f"Identity evolution: {evolution}")
        else:
            logger.warning("Agent not wrapped with DTESN identity")
    
    await az.shutdown()
    logger.info("=== DTESN Example Complete ===")


async def full_integration_example():
    """Example using all integration features together."""
    logger.info("=== Full Integration Example ===")
    
    # Enable all features
    config = AphroditechoZeroConfig(
        enable_opencog=True,
        enable_ecan=True,
        enable_hypergraph_ql=True,
        enable_dtesn=True,
        enable_membrane_computing=True,
        enable_echo_self=True,
        enable_aar=True,
        enable_arena_simulation=True,
        enable_relation_graphs=True,
        enable_asmoses=True,
        max_concurrent_agents=10
    )
    
    az = AphroditechoZero(config)
    await az.initialize()
    
    # Create a fully-integrated agent
    logger.info("Creating fully-integrated agent...")
    agent = await az.create_agent(
        agent_id="fully-integrated",
        name="IntegratedAgent",
        agent_type="hybrid"
    )
    
    # Process a complex request
    logger.info("Processing complex request...")
    result = await az.process_inference_request(
        agent_id="fully-integrated",
        prompt="""Analyze how the integration of agent-zero, OpenCog atomspace,
                  Deep Tree Echo membrane computing, and AAR orchestration
                  creates a novel cognitive architecture.""",
        context={
            'require_opencog': True,
            'require_dtesn': True,
            'require_aar': True,
            'detail_level': 'comprehensive'
        }
    )
    
    logger.info(f"Response: {result['response'][:300]}...")
    logger.info(f"Integration metadata: {result['metadata']}")
    
    # Demonstrate all subsystems working together
    logger.info("\n=== Subsystem Status ===")
    logger.info(f"OpenCog Bridge: {'Active' if az.opencog_bridge else 'Inactive'}")
    logger.info(f"DTESN Identity: {'Active' if az.dtesn_identity else 'Inactive'}")
    logger.info(f"AAR Orchestrator: {'Active' if az.aar_orchestrator else 'Inactive'}")
    
    await az.shutdown()
    logger.info("=== Full Integration Example Complete ===")


async def main():
    """Run all examples."""
    logger.info("Starting Aphroditecho-Zero Examples\n")
    
    try:
        # Run basic example
        await basic_example()
        await asyncio.sleep(1)
        
        # Run multi-agent example
        await multi_agent_example()
        await asyncio.sleep(1)
        
        # Run OpenCog integration example
        await opencog_integration_example()
        await asyncio.sleep(1)
        
        # Run DTESN example
        await dtesn_example()
        await asyncio.sleep(1)
        
        # Run full integration example
        await full_integration_example()
        
        logger.info("\n=== All Examples Completed Successfully ===")
        
    except Exception as e:
        logger.error(f"Error running examples: {e}", exc_info=True)
        return 1
        
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
