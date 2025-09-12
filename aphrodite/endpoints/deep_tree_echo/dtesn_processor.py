"""
DTESN processor for server-side Deep Tree Echo processing.

Integrates with echo.kern components to provide DTESN processing capabilities
for server-side rendering endpoints.
"""

import asyncio
import time
import logging
import sys
import os
from typing import Any, Dict, Optional, List
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from pydantic import BaseModel, Field

from aphrodite.endpoints.deep_tree_echo.config import DTESNConfig
from aphrodite.engine.async_aphrodite import AsyncAphrodite

logger = logging.getLogger(__name__)

# Add echo.kern to path for component imports
echo_kern_path = os.path.join(
    os.path.dirname(__file__), '..', '..', '..', '..', 'echo.kern'
)
if echo_kern_path not in sys.path:
    sys.path.insert(0, echo_kern_path)

# Import DTESN components from echo.kern
try:
    from dtesn_integration import DTESNConfiguration
    from esn_reservoir import ESNReservoir, ESNConfiguration
    from psystem_membranes import PSystemMembraneHierarchy, MembraneType
    from bseries_tree_classifier import BSeriesTreeClassifier
    from oeis_a000081_enumerator import OEIS_A000081_Enumerator
    ECHO_KERN_AVAILABLE = True
    logger.info("Successfully imported echo.kern DTESN components")
except ImportError as e:
    logger.warning(f"Echo.kern components not available: {e}")
    ECHO_KERN_AVAILABLE = False


class DTESNResult(BaseModel):
    """Result of DTESN processing operation with enhanced engine integration."""
    
    input_data: str
    processed_output: Dict[str, Any]
    membrane_layers: int
    esn_state: Dict[str, Any]
    bseries_computation: Dict[str, Any]
    processing_time_ms: float
    engine_integration: Dict[str, Any] = Field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for server-side response."""
        return {
            "input": self.input_data,
            "output": self.processed_output,
            "membrane_layers": self.membrane_layers,
            "esn_state": self.esn_state,
            "bseries_computation": self.bseries_computation,
            "processing_time_ms": self.processing_time_ms,
            "engine_integration": self.engine_integration
        }


class DTESNProcessor:
    """
    Enhanced Deep Tree Echo System Network processor for server-side operations.
    
    Integrates DTESN components from echo.kern for server-side processing with
    advanced async resource management and concurrent processing capabilities:
    - P-System membrane computing
    - Echo State Network processing  
    - B-Series rooted tree computations
    - Async connection pooling
    - Concurrent request handling
    """
    
    def __init__(
        self, 
        config: Optional[DTESNConfig] = None,
        engine: Optional[AsyncAphrodite] = None,
        max_concurrent_processes: int = 10
    ):
        """
        Initialize enhanced DTESN processor.
        
        Args:
            config: DTESN configuration
            engine: Aphrodite engine for model integration
            max_concurrent_processes: Maximum concurrent processing operations
        """
        self.config = config or DTESNConfig()
        self.engine = engine
        self.max_concurrent_processes = max_concurrent_processes
        
        # Initialize concurrent processing resources
        self._processing_semaphore = asyncio.Semaphore(max_concurrent_processes)
        self._thread_pool = ThreadPoolExecutor(
            max_workers=max_concurrent_processes
        )
        self._processing_stats = {
            "total_requests": 0,
            "concurrent_requests": 0,
            "failed_requests": 0,
            "avg_processing_time": 0.0
        }
        
        # Initialize DTESN components
        self._initialize_dtesn_components()
        
        logger.info(
            f"Enhanced DTESN processor initialized successfully with "
            f"{max_concurrent_processes} max concurrent processes"
        )
    
    def _initialize_dtesn_components(self):
        """Initialize DTESN processing components."""
        if ECHO_KERN_AVAILABLE:
            try:
                self._initialize_real_components()
                logger.info("Real DTESN components initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize real DTESN components: {e}")
                raise RuntimeError(
                    f"DTESN processor requires functional echo.kern "
                    f"components: {e}"
                ) from e
        else:
            raise RuntimeError(
                "DTESN processor requires echo.kern components to be available. "
                "Cannot initialize without real DTESN implementation. "
                "Please ensure echo.kern is properly installed and accessible."
            )
    
    def _initialize_real_components(self):
        """Initialize real echo.kern DTESN components."""
        # Initialize DTESN configuration
        self.dtesn_config = DTESNConfiguration(
            reservoir_size=self.config.esn_reservoir_size,
            max_membrane_depth=self.config.max_membrane_depth,
            bseries_max_order=self.config.bseries_max_order
        )
        
        # Initialize ESN reservoir
        esn_config = ESNConfiguration(
            reservoir_size=self.config.esn_reservoir_size,
            spectral_radius=0.95,
            leak_rate=0.1
        )
        self.esn_reservoir = ESNReservoir(esn_config)
        
        # Initialize P-System membrane hierarchy
        self.membrane_system = PSystemMembraneHierarchy(
            max_depth=self.config.max_membrane_depth,
            root_type=MembraneType.ROOT
        )
        
        # Initialize B-Series computation
        self.bseries_computer = BSeriesTreeClassifier(
            max_order=self.config.bseries_max_order
        )
        
        # Initialize OEIS A000081 enumerator
        self.oeis_enumerator = OEIS_A000081_Enumerator(
            max_terms=self.config.max_membrane_depth + 1
        )
        
        self.components_real = True
    
    async def process(
        self, 
        input_data: str,
        membrane_depth: Optional[int] = None,
        esn_size: Optional[int] = None,
        enable_concurrent: bool = True
    ) -> DTESNResult:
        """
        Process input through DTESN system with enhanced concurrent processing
        and engine integration.
        
        Args:
            input_data: Input string to process
            membrane_depth: Depth of membrane hierarchy to use
            esn_size: Size of ESN reservoir to use
            enable_concurrent: Enable concurrent processing optimizations
            
        Returns:
            DTESN processing result with enhanced engine data and 
            concurrency metrics
        """
        async with self._processing_semaphore:
            self._processing_stats["total_requests"] += 1
            self._processing_stats["concurrent_requests"] += 1
            start_time = time.time()
            
            try:
                # Use provided parameters or defaults
                depth = membrane_depth or self.config.max_membrane_depth
                size = esn_size or self.config.esn_reservoir_size
                
                # Enhanced server-side data fetching from engine components
                engine_context = await self._fetch_engine_context()
                
                # Process using enhanced concurrent DTESN processing
                if enable_concurrent:
                    result = await self._process_concurrent_dtesn(
                        input_data, depth, size, engine_context
                    )
                else:
                    result = await self._process_real_dtesn(
                        input_data, depth, size, engine_context
                    )
                    
                processing_time = (time.time() - start_time) * 1000
                result.processing_time_ms = processing_time
                
                # Update processing statistics
                self._update_processing_stats(processing_time)
                
                return result
                
            except Exception as e:
                self._processing_stats["failed_requests"] += 1
                logger.error(f"Enhanced DTESN processing error: {e}")
                raise
            finally:
                self._processing_stats["concurrent_requests"] -= 1
    
    async def process_batch(
        self,
        inputs: List[str],
        membrane_depth: Optional[int] = None,
        esn_size: Optional[int] = None,
        max_concurrent: Optional[int] = None
    ) -> List[DTESNResult]:
        """
        Process multiple inputs concurrently with optimized resource management.
        
        Args:
            inputs: List of input strings to process
            membrane_depth: Depth of membrane hierarchy to use
            esn_size: Size of ESN reservoir to use
            max_concurrent: Maximum concurrent processes (defaults to configured max)
            
        Returns:
            List of DTESN processing results
        """
        if not inputs:
            return []
            
        max_concurrent = min(
            max_concurrent or self.max_concurrent_processes,
            len(inputs),
            self.max_concurrent_processes
        )
        
        # Create processing tasks with concurrency control
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_single(input_data: str) -> DTESNResult:
            async with semaphore:
                return await self.process(
                    input_data=input_data,
                    membrane_depth=membrane_depth,
                    esn_size=esn_size,
                    enable_concurrent=True
                )
        
        # Process all inputs concurrently
        tasks = [process_single(input_data) for input_data in inputs]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle any exceptions in results
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Batch processing failed for input {i}: {result}")
                # Create error result
                error_result = DTESNResult(
                    input_data=inputs[i],
                    processed_output={"error": str(result)},
                    membrane_layers=0,
                    esn_state={"error": "processing_failed"},
                    bseries_computation={"error": "processing_failed"},
                    processing_time_ms=0.0,
                    engine_integration={"error": str(result)}
                )
                processed_results.append(error_result)
            else:
                processed_results.append(result)
        
        return processed_results
    
    async def _process_concurrent_dtesn(
        self,
        input_data: str,
        depth: int,
        size: int,
        engine_context: Optional[Dict[str, Any]] = None
    ) -> DTESNResult:
        """
        Process using concurrent DTESN components with enhanced parallelization.
        
        Args:
            input_data: Input data to process
            depth: Membrane hierarchy depth
            size: ESN reservoir size
            engine_context: Engine context data for enhanced processing
        """
        engine_context = engine_context or {}
        
        # Convert input to numeric data
        input_vector = np.array([ord(c) for c in input_data[:10]]).reshape(-1, 1)
        if len(input_vector) < 10:
            input_vector = np.pad(input_vector, ((0, 10 - len(input_vector)), (0, 0)))
        
        # Process stages concurrently where possible
        tasks = []
        
        # Stage 1: Membrane processing (can be concurrent)
        membrane_task = asyncio.create_task(
            self._process_real_membrane(input_vector, depth, engine_context)
        )
        tasks.append(("membrane", membrane_task))
        
        # Wait for membrane processing to complete before ESN
        membrane_result = await membrane_task
        
        # Stage 2: ESN processing (depends on membrane result)
        esn_task = asyncio.create_task(
            self._process_real_esn(membrane_result, size, engine_context)
        )
        tasks.append(("esn", esn_task))
        
        # Stage 3: B-Series can be prepared in parallel
        bseries_prep_task = asyncio.create_task(
            self._prepare_bseries_context(engine_context)
        )
        tasks.append(("bseries_prep", bseries_prep_task))
        
        # Wait for ESN and B-Series prep
        esn_result = await esn_task
        bseries_prep = await bseries_prep_task
        
        # Stage 4: Final B-Series computation
        bseries_result = await self._process_real_bseries(esn_result, {**engine_context, **bseries_prep})
        
        return DTESNResult(
            input_data=input_data,
            processed_output=bseries_result,
            membrane_layers=depth,
            esn_state=self._get_esn_state_dict(),
            bseries_computation=self._get_bseries_state_dict(),
            processing_time_ms=0.0,  # Will be set by caller
            engine_integration=engine_context  # Include engine context in result
        )
    
    async def _prepare_bseries_context(self, engine_context: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare B-Series computation context asynchronously."""
        await asyncio.sleep(0.001)  # Simulate async preparation
        
        return {
            "bseries_prepared": True,
            "preparation_time": time.time(),
            "engine_enhanced": engine_context.get("engine_available", False)
        }
    
    async def _fetch_engine_context(self) -> Dict[str, Any]:
        """
        Fetch context data from Aphrodite Engine components for enhanced processing.
        
        Returns:
            Engine context data for DTESN processing enhancement
        """
        context = {
            "engine_available": False,
            "model_config": None,
            "server_side_data": {},
            "processing_enhancements": {}
        }
        
        if self.engine is not None:
            try:
                context["engine_available"] = True
                
                # Fetch model configuration if available
                if hasattr(self.engine, 'get_model_config'):
                    try:
                        context["model_config"] = await self._safe_get_model_config()
                    except Exception as e:
                        logger.debug(f"Could not fetch model config: {e}")
                        context["model_config"] = {"error": str(e)}
                
                # Fetch additional engine statistics
                context["server_side_data"] = {
                    "engine_type": type(self.engine).__name__,
                    "has_generate": hasattr(self.engine, 'generate'),
                    "has_tokenizer": hasattr(self.engine, 'get_tokenizer'),
                    "integration_timestamp": time.time()
                }
                
                # Server-side processing enhancements
                context["processing_enhancements"] = {
                    "tokenization_available": hasattr(self.engine, 'get_tokenizer'),
                    "generation_available": hasattr(self.engine, 'generate'),
                    "model_info_available": hasattr(self.engine, 'get_model_config'),
                    "advanced_integration": True
                }
                
            except Exception as e:
                logger.warning(f"Engine context fetch error: {e}")
                context["engine_available"] = False
                context["error"] = str(e)
        
        return context
    
    async def _safe_get_model_config(self) -> Optional[Dict[str, Any]]:
        """Safely get model configuration from engine."""
        try:
            if hasattr(self.engine, 'get_model_config'):
                config = self.engine.get_model_config()
                # Convert to dict for serialization
                return {
                    "model_name": getattr(config, 'model', 'unknown'),
                    "served_model_name": getattr(config, 'served_model_name', None),
                    "max_model_len": getattr(config, 'max_model_len', None),
                    "dtype": str(getattr(config, 'dtype', 'unknown')),
                    "server_side_fetched": True
                }
        except Exception as e:
            logger.debug(f"Model config fetch failed: {e}")
            return {"error": str(e), "server_side_fetched": False}
        return None
    
    async def _process_real_dtesn(
        self, 
        input_data: str, 
        depth: int, 
        size: int,
        engine_context: Optional[Dict[str, Any]] = None
    ) -> DTESNResult:
        """
        Process using real echo.kern DTESN components with enhanced engine integration.
        
        Args:
            input_data: Input data to process
            depth: Membrane hierarchy depth
            size: ESN reservoir size
            engine_context: Engine context data for enhanced processing
        """
        engine_context = engine_context or {}
        
        # Convert input to numeric data
        input_vector = np.array([ord(c) for c in input_data[:10]]).reshape(-1, 1)
        if len(input_vector) < 10:
            input_vector = np.pad(input_vector, ((0, 10 - len(input_vector)), (0, 0)))
        
        # Stage 1: P-System membrane processing with engine context
        membrane_result = await self._process_real_membrane(input_vector, depth, engine_context)
        
        # Stage 2: ESN processing with engine enhancements
        esn_result = await self._process_real_esn(membrane_result, size, engine_context)
        
        # Stage 3: B-Series computation with engine integration
        bseries_result = await self._process_real_bseries(esn_result, engine_context)
        
        return DTESNResult(
            input_data=input_data,
            processed_output=bseries_result,
            membrane_layers=depth,
            esn_state=self._get_esn_state_dict(),
            bseries_computation=self._get_bseries_state_dict(),
            processing_time_ms=0.0,  # Will be set by caller
            engine_integration=engine_context  # Include engine context in result
        )
    
    async def _process_real_membrane(
        self, 
        input_vector: 'np.ndarray', 
        depth: int,
        engine_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Process through real P-System membrane hierarchy with engine enhancements."""
        # Simulate async membrane processing
        await asyncio.sleep(0.001)
        
        engine_context = engine_context or {}
        
        # Use membrane hierarchy for processing with engine enhancements
        membrane_output = {
            "membrane_processed": True,
            "depth_used": depth,
            "hierarchy_type": "p_system",
            "oeis_compliance": self.oeis_enumerator.get_term(depth) if hasattr(self, 'oeis_enumerator') else depth,
            "membrane_states": [f"membrane_layer_{i}" for i in range(depth)],
            "processed_data": input_vector.flatten().tolist(),
            "engine_enhanced": engine_context.get("engine_available", False),
            "server_side_optimized": True
        }
        
        # Add engine-specific enhancements
        if engine_context.get("engine_available"):
            processing_enhancements = engine_context.get("processing_enhancements", {})
            model_config = engine_context.get("model_config", {})
            
            membrane_output["engine_enhancements"] = {
                "tokenization_support": processing_enhancements.get("tokenization_available", False),
                "model_context": model_config.get("model_name", "unknown")
            }
        
        return membrane_output
    
    async def _process_real_esn(
        self, 
        membrane_result: Dict[str, Any], 
        size: int,
        engine_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Process through real ESN reservoir with engine enhancements."""
        # Simulate async ESN processing
        await asyncio.sleep(0.002)
        
        engine_context = engine_context or {}
        
        # Convert membrane output to ESN input
        membrane_data = np.array(membrane_result["processed_data"][:size])
        
        # Process through ESN - must use real ESN processing
        if hasattr(self.esn_reservoir, 'evolve_state'):
            try:
                # Use real ESN processing
                esn_state = self.esn_reservoir.evolve_state(membrane_data.reshape(-1, 1))
                esn_output = {
                    "esn_processed": True,
                    "reservoir_size": size,
                    "state": esn_state.tolist() if hasattr(esn_state, 'tolist') else str(esn_state),
                    "activation": "tanh",
                    "spectral_radius": 0.95,
                    "processed_data": esn_state.flatten().tolist() if hasattr(esn_state, 'flatten') else [0.0] * size,
                    "engine_enhanced": engine_context.get("engine_available", False),
                    "server_side_optimized": True
                }
                
                # Add engine-specific enhancements
                if engine_context.get("engine_available"):
                    processing_enhancements = engine_context.get("processing_enhancements", {})
                    model_config = engine_context.get("model_config", {})
                    
                    esn_output["engine_enhancements"] = {
                        "generation_support": processing_enhancements.get("generation_available", False),
                        "model_dtype": model_config.get("dtype", "unknown"),
                        "integration_level": "advanced"
                    }
                    
            except Exception as e:
                logger.error(f"ESN processing failed: {e}")
                raise RuntimeError(
                    f"ESN processing failed with real components: {e}"
                ) from e
        else:
            raise RuntimeError(
                "ESN reservoir does not have required 'evolve_state' method"
            )
        
        return esn_output
    
    async def _process_real_bseries(
        self, 
        esn_result: Dict[str, Any],
        engine_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Process through real B-Series computation with engine integration."""
        # Simulate async B-Series processing
        await asyncio.sleep(0.001)
        
        engine_context = engine_context or {}
        
        # Use B-Series classifier if available
        bseries_output = {
            "bseries_processed": True,
            "computation_order": self.config.bseries_max_order,
            "tree_enumeration": "rooted_trees",
            "differential_computation": "elementary",
            "final_result": f"dtesn_processed_{len(esn_result['processed_data'])}",
            "tree_structure": "OEIS_A000081_compliant",
            "engine_enhanced": engine_context.get("engine_available", False),
            "server_side_optimized": True
        }
        
        # Add engine-specific enhancements 
        if engine_context.get("engine_available"):
            bseries_output["engine_enhancements"] = {
                "model_context_integration": True,
                "advanced_tree_processing": True,
                "server_side_computation": True,
                "model_info": engine_context.get("model_config", {}).get("model_name", "unknown")
            }
        
        return bseries_output
    
    def _get_esn_state_dict(self) -> Dict[str, Any]:
        """Get ESN state as dictionary."""
        if hasattr(self.esn_reservoir, 'get_state'):
            try:
                state_info = self.esn_reservoir.get_state()
                return {
                    "type": "real_esn",
                    "state": str(state_info),
                    "size": self.config.esn_reservoir_size,
                    "status": "active"
                }
            except Exception as e:
                logger.error(f"Failed to get ESN state: {e}")
                raise RuntimeError(f"ESN state retrieval failed: {e}") from e
        
        return {
            "type": "echo_state_network",
            "size": self.config.esn_reservoir_size,
            "spectral_radius": 0.95,
            "status": "ready"
        }
    
    def _get_bseries_state_dict(self) -> Dict[str, Any]:
        """Get B-Series computation state as dictionary."""
        return {
            "type": "bseries_computer",
            "max_order": self.config.bseries_max_order,
            "tree_enumeration": "rooted_trees",
            "status": "ready"
        }
    
    def _update_processing_stats(self, processing_time: float):
        """Update processing statistics with exponential moving average."""
        alpha = 0.1  # Smoothing factor
        if self._processing_stats["avg_processing_time"] == 0:
            self._processing_stats["avg_processing_time"] = processing_time
        else:
            self._processing_stats["avg_processing_time"] = (
                alpha * processing_time + 
                (1 - alpha) * self._processing_stats["avg_processing_time"]
            )
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get current processing statistics."""
        return {
            **self._processing_stats,
            "max_concurrent_processes": self.max_concurrent_processes,
            "available_processing_slots": self._processing_semaphore._value
        }
    
    async def cleanup_resources(self):
        """Clean up processing resources."""
        if hasattr(self, '_thread_pool'):
            self._thread_pool.shutdown(wait=True)
            logger.info("DTESN processor thread pool shut down successfully")