#!/usr/bin/env python3
"""
Tests for Aphroditecho-Zero Integration

Tests the integration of agent-zero with Aphrodite Engine, OpenCog,
Deep Tree Echo, and AAR orchestration.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock

from mods.agentzero import (
    AphroditechoZero,
    AphroditechoZeroConfig,
    AgentZeroAdapter,
    OpenCogAgentBridge,
    DeepTreeEchoIdentity,
    AARAgentOrchestrator,
)


class TestAphroditechoZeroConfig:
    """Test configuration class."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = AphroditechoZeroConfig()
        
        assert config.enable_opencog is True
        assert config.enable_dtesn is True
        assert config.enable_aar is True
        assert config.max_concurrent_agents == 10
        
    def test_custom_config(self):
        """Test custom configuration."""
        config = AphroditechoZeroConfig(
            enable_opencog=False,
            max_concurrent_agents=20,
            memory_limit_mb=8192
        )
        
        assert config.enable_opencog is False
        assert config.max_concurrent_agents == 20
        assert config.memory_limit_mb == 8192


class TestAgentZeroAdapter:
    """Test agent-zero adapter."""
    
    @pytest.fixture
    async def adapter(self):
        """Create adapter instance."""
        config = AphroditechoZeroConfig()
        adapter = AgentZeroAdapter(config)
        yield adapter
        if adapter.initialized:
            await adapter.shutdown()
    
    @pytest.mark.asyncio
    async def test_initialization(self, adapter):
        """Test adapter initialization."""
        await adapter.initialize()
        
        assert adapter.initialized is True
        
    @pytest.mark.asyncio
    async def test_create_agent(self, adapter):
        """Test agent creation."""
        await adapter.initialize()
        
        # Create an agent (will work in standalone mode if agent-zero not available)
        agent_id = "test-agent-001"
        
        # This may return None if agent-zero is not installed
        agent = await adapter.create_agent(agent_id, "TestAgent")
        
        # Check that agent_id is tracked
        if agent is not None:
            assert agent_id in adapter.agents
            assert agent_id in adapter.agent_contexts
            
    @pytest.mark.asyncio
    async def test_shutdown(self, adapter):
        """Test adapter shutdown."""
        await adapter.initialize()
        await adapter.shutdown()
        
        assert adapter.initialized is False
        assert len(adapter.agents) == 0


class TestOpenCogAgentBridge:
    """Test OpenCog integration bridge."""
    
    @pytest.fixture
    async def bridge(self):
        """Create bridge instance."""
        config = AphroditechoZeroConfig()
        bridge = OpenCogAgentBridge(config)
        yield bridge
        if bridge.initialized:
            await bridge.shutdown()
    
    @pytest.mark.asyncio
    async def test_initialization(self, bridge):
        """Test bridge initialization."""
        await bridge.initialize()
        
        # Bridge may initialize even if OpenCog is not available
        assert bridge.initialized is True
        
    @pytest.mark.asyncio
    async def test_register_agent(self, bridge):
        """Test agent registration."""
        await bridge.initialize()
        
        mock_agent = Mock()
        await bridge.register_agent("test-agent", mock_agent)
        
        # Should complete without error
        if bridge.atomspace:
            assert "test-agent" in bridge.agent_atoms
            
    @pytest.mark.asyncio
    async def test_enhance_reasoning(self, bridge):
        """Test reasoning enhancement."""
        await bridge.initialize()
        
        mock_agent = Mock()
        await bridge.register_agent("reasoning-agent", mock_agent)
        
        result = await bridge.enhance_agent_reasoning(
            agent_id="reasoning-agent",
            query="Test query",
            context={'test': True}
        )
        
        assert isinstance(result, dict)
        assert 'enhanced' in result


class TestDeepTreeEchoIdentity:
    """Test Deep Tree Echo identity integration."""
    
    @pytest.fixture
    async def identity(self):
        """Create identity instance."""
        config = AphroditechoZeroConfig()
        identity = DeepTreeEchoIdentity(config)
        yield identity
        if identity.initialized:
            await identity.shutdown()
    
    @pytest.mark.asyncio
    async def test_initialization(self, identity):
        """Test identity initialization."""
        await identity.initialize()
        
        assert identity.initialized is True
        
    @pytest.mark.asyncio
    async def test_wrap_agent(self, identity):
        """Test agent wrapping with DTESN."""
        await identity.initialize()
        
        mock_agent = Mock()
        wrapper = await identity.wrap_agent("dtesn-test", mock_agent)
        
        assert wrapper is not None
        assert "dtesn-test" in identity.wrapped_agents
        assert wrapper.agent_id == "dtesn-test"
        
    @pytest.mark.asyncio
    async def test_evolve_agent_identity(self, identity):
        """Test agent identity evolution."""
        await identity.initialize()
        
        mock_agent = Mock()
        await identity.wrap_agent("evolve-test", mock_agent)
        
        result = await identity.evolve_agent_identity(
            agent_id="evolve-test",
            performance_data={'accuracy': 0.9}
        )
        
        assert isinstance(result, dict)
        assert result['evolved'] is True


class TestAARAgentOrchestrator:
    """Test AAR orchestration system."""
    
    @pytest.fixture
    async def orchestrator(self):
        """Create orchestrator instance."""
        config = AphroditechoZeroConfig()
        orchestrator = AARAgentOrchestrator(config)
        yield orchestrator
        if orchestrator.initialized:
            await orchestrator.shutdown()
    
    @pytest.mark.asyncio
    async def test_initialization(self, orchestrator):
        """Test orchestrator initialization."""
        await orchestrator.initialize()
        
        assert orchestrator.initialized is True
        assert orchestrator.lifecycle_manager is not None
        assert orchestrator.arena_simulator is not None
        assert orchestrator.relation_graph is not None
        
    @pytest.mark.asyncio
    async def test_register_agent(self, orchestrator):
        """Test agent registration."""
        await orchestrator.initialize()
        
        mock_agent = Mock()
        await orchestrator.register_agent("aar-test", mock_agent)
        
        assert "aar-test" in orchestrator.lifecycle_manager.agents
        assert "aar-test" in orchestrator.relation_graph.graph
        
    @pytest.mark.asyncio
    async def test_multi_agent_coordination(self, orchestrator):
        """Test multi-agent task coordination."""
        await orchestrator.initialize()
        
        mock_agent = Mock()
        await orchestrator.register_agent("coordinator", mock_agent)
        
        result = await orchestrator.coordinate_multi_agent_task(
            root_agent_id="coordinator",
            task_description="Test task",
            max_agents=5
        )
        
        assert isinstance(result, dict)
        assert result['success'] is True
        assert 'hierarchy' in result


class TestAphroditechoZero:
    """Test main integration class."""
    
    @pytest.fixture
    async def aphroditecho_zero(self):
        """Create Aphroditecho-Zero instance."""
        config = AphroditechoZeroConfig(
            enable_opencog=True,
            enable_dtesn=True,
            enable_aar=True
        )
        az = AphroditechoZero(config)
        yield az
        if az.initialized:
            await az.shutdown()
    
    @pytest.mark.asyncio
    async def test_initialization(self, aphroditecho_zero):
        """Test Aphroditecho-Zero initialization."""
        await aphroditecho_zero.initialize()
        
        assert aphroditecho_zero.initialized is True
        assert aphroditecho_zero.agent_adapter is not None
        
    @pytest.mark.asyncio
    async def test_create_agent(self, aphroditecho_zero):
        """Test integrated agent creation."""
        await aphroditecho_zero.initialize()
        
        agent = await aphroditecho_zero.create_agent(
            agent_id="integration-test",
            name="IntegrationTestAgent"
        )
        
        # Agent may be None if agent-zero is not installed
        # But the call should not raise an error
        
    @pytest.mark.asyncio
    async def test_process_inference(self, aphroditecho_zero):
        """Test inference request processing."""
        await aphroditecho_zero.initialize()
        
        await aphroditecho_zero.create_agent(
            agent_id="inference-test",
            name="InferenceAgent"
        )
        
        result = await aphroditecho_zero.process_inference_request(
            agent_id="inference-test",
            prompt="Test prompt",
            context={'test': True}
        )
        
        assert isinstance(result, dict)
        assert 'agent_id' in result
        assert 'prompt' in result
        assert 'response' in result
        assert 'metadata' in result
        
    @pytest.mark.asyncio
    async def test_full_integration(self, aphroditecho_zero):
        """Test full integration with all subsystems."""
        await aphroditecho_zero.initialize()
        
        # Create agent with all integrations
        agent = await aphroditecho_zero.create_agent(
            agent_id="full-integration-test",
            name="FullIntegrationAgent",
            agent_type="hybrid"
        )
        
        # Verify subsystems are initialized
        assert aphroditecho_zero.agent_adapter is not None
        
        # OpenCog may or may not be available
        if aphroditecho_zero.opencog_bridge:
            assert aphroditecho_zero.opencog_bridge.initialized is True
            
        # DTESN may or may not be available
        if aphroditecho_zero.dtesn_identity:
            assert aphroditecho_zero.dtesn_identity.initialized is True
            
        # AAR should be available
        if aphroditecho_zero.aar_orchestrator:
            assert aphroditecho_zero.aar_orchestrator.initialized is True
            
    @pytest.mark.asyncio
    async def test_shutdown(self, aphroditecho_zero):
        """Test shutdown procedure."""
        await aphroditecho_zero.initialize()
        await aphroditecho_zero.create_agent("shutdown-test", "ShutdownAgent")
        
        await aphroditecho_zero.shutdown()
        
        assert aphroditecho_zero.initialized is False


class TestIntegrationScenarios:
    """Integration scenario tests."""
    
    @pytest.mark.asyncio
    async def test_multi_agent_scenario(self):
        """Test multi-agent coordination scenario."""
        config = AphroditechoZeroConfig(
            enable_aar=True,
            max_concurrent_agents=5
        )
        
        az = AphroditechoZero(config)
        await az.initialize()
        
        try:
            # Create root agent
            await az.create_agent("root", "RootAgent")
            
            # Create subordinate agents
            for i in range(3):
                await az.create_agent(
                    f"worker-{i}",
                    f"Worker{i}",
                    config={'superior': 'root'}
                )
            
            # Test coordination
            if az.aar_orchestrator:
                result = await az.aar_orchestrator.coordinate_multi_agent_task(
                    root_agent_id="root",
                    task_description="Test multi-agent task",
                    max_agents=5
                )
                
                assert result['success'] is True
                
        finally:
            await az.shutdown()
            
    @pytest.mark.asyncio
    async def test_knowledge_persistence_scenario(self):
        """Test knowledge persistence through OpenCog."""
        config = AphroditechoZeroConfig(
            enable_opencog=True,
            enable_hypergraph_ql=True
        )
        
        az = AphroditechoZero(config)
        await az.initialize()
        
        try:
            await az.create_agent("knowledge-agent", "KnowledgeAgent")
            
            if az.opencog_bridge and az.opencog_bridge.hypergraph_ql:
                # Store knowledge
                await az.opencog_bridge.hypergraph_ql.store_agent_memory(
                    agent_id="knowledge-agent",
                    memory_type="semantic",
                    content="Test knowledge",
                    metadata={'test': True}
                )
                
                # Should not raise error
                
        finally:
            await az.shutdown()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
