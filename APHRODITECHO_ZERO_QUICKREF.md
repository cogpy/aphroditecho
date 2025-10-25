# Aphroditecho-Zero Quick Reference

## Installation

```bash
pip install numpy networkx
```

## Basic Usage

```python
import asyncio
from mods.agentzero import AphroditechoZero, AphroditechoZeroConfig

async def main():
    # Initialize
    az = AphroditechoZero(AphroditechoZeroConfig())
    await az.initialize()
    
    # Create agent
    agent = await az.create_agent("agent-001", "MyAgent")
    
    # Process request
    result = await az.process_inference_request(
        "agent-001",
        "Your prompt here"
    )
    
    print(result['response'])
    
    # Cleanup
    await az.shutdown()

asyncio.run(main())
```

## Configuration Options

```python
config = AphroditechoZeroConfig(
    # Agent-zero
    enable_multi_agent=True,
    max_concurrent_agents=10,
    
    # OpenCog
    enable_opencog=True,
    enable_ecan=True,
    enable_hypergraph_ql=True,
    
    # Deep Tree Echo
    enable_dtesn=True,
    enable_membrane_computing=True,
    enable_echo_self=True,
    
    # AAR
    enable_aar=True,
    enable_arena_simulation=True,
    enable_relation_graphs=True,
)
```

## Common Patterns

### Multi-Agent Coordination

```python
# Create coordinator
await az.create_agent("coordinator", "Coordinator")

# Create workers
for i in range(3):
    await az.create_agent(
        f"worker-{i}",
        f"Worker{i}",
        config={'superior': 'coordinator'}
    )

# Coordinate task
result = await az.aar_orchestrator.coordinate_multi_agent_task(
    "coordinator",
    "Task description"
)
```

### Knowledge Storage

```python
if az.opencog_bridge and az.opencog_bridge.hypergraph_ql:
    await az.opencog_bridge.hypergraph_ql.store_agent_memory(
        agent_id="agent-001",
        memory_type="semantic",
        content="Knowledge to store",
        metadata={'importance': 0.9}
    )
```

### DTESN Processing

```python
if az.dtesn_identity and "agent-001" in az.dtesn_identity.wrapped_agents:
    wrapper = az.dtesn_identity.wrapped_agents["agent-001"]
    result = await wrapper.process_with_dtesn(
        input_data="Data to process",
        context={'temporal': True}
    )
```

## Component Status Check

```python
print(f"Agent Adapter: {az.agent_adapter is not None}")
print(f"OpenCog: {az.opencog_bridge is not None}")
print(f"DTESN: {az.dtesn_identity is not None}")
print(f"AAR: {az.aar_orchestrator is not None}")
```

## Troubleshooting

### No module named 'agent'
```python
# Use standalone mode
config = AphroditechoZeroConfig(enable_multi_agent=False)
```

### Missing dependencies
```bash
pip install numpy networkx
```

### Disable optional components
```python
config = AphroditechoZeroConfig(
    enable_opencog=False,  # Disable OpenCog
    enable_dtesn=False,    # Disable DTESN
)
```

## File Locations

- **Module**: `mods/agentzero/`
- **Examples**: `examples/aphroditecho_zero_demo.py`
- **Tests**: `test_aphroditecho_zero.py`
- **Docs**: `APHRODITECHO_ZERO_INTEGRATION_GUIDE.md`

## Key Classes

- `AphroditechoZero` - Main integration class
- `AphroditechoZeroConfig` - Configuration
- `AgentZeroAdapter` - Agent-zero wrapper
- `OpenCogAgentBridge` - OpenCog integration
- `DeepTreeEchoIdentity` - DTESN integration
- `AARAgentOrchestrator` - AAR orchestration

## API Reference

### AphroditechoZero

```python
async def initialize()
async def create_agent(agent_id, name, agent_type, config)
async def process_inference_request(agent_id, prompt, context)
async def shutdown()
```

### AARAgentOrchestrator

```python
async def register_agent(agent_id, agent, superior_id)
async def coordinate_multi_agent_task(root_agent_id, task, max_agents)
```

### OpenCogAgentBridge

```python
async def register_agent(agent_id, agent)
async def enhance_agent_reasoning(agent_id, query, context)
async def evolve_agent_architecture(agent_id, performance_metrics)
```

### DeepTreeEchoIdentity

```python
async def wrap_agent(agent_id, agent)
async def evolve_agent_identity(agent_id, performance_data)
async def synchronize_echo_state(agent_id, global_echo_state)
```

## Version

**Current**: v0.1.0 - Initial Release
**Status**: Production Ready ✅

## Support

- GitHub Issues
- Documentation: `mods/agentzero/README.md`
- Examples: `examples/aphroditecho_zero_demo.py`
