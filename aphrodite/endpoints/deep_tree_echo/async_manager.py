"""
Async resource management for Deep Tree Echo server-side processing.

Provides connection pooling, resource management, and concurrency control
for efficient async server-side request handling.
"""

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Dict, List, Optional, Set
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class ConnectionPoolConfig:
    """Configuration for async connection pooling."""
    
    max_connections: int = 100
    min_connections: int = 10
    connection_timeout: float = 30.0
    idle_timeout: float = 300.0
    max_retries: int = 3
    retry_delay: float = 0.1


@dataclass 
class ResourcePoolStats:
    """Statistics for resource pool monitoring."""
    
    active_connections: int = 0
    idle_connections: int = 0
    total_requests: int = 0
    failed_requests: int = 0
    avg_response_time: float = 0.0
    pool_utilization: float = 0.0
    last_updated: float = field(default_factory=time.time)


class AsyncConnectionPool:
    """
    Async connection pool for efficient resource management.
    
    Manages connections for DTESN processing requests with proper
    lifecycle management and resource limiting.
    """
    
    def __init__(self, config: Optional[ConnectionPoolConfig] = None):
        """Initialize async connection pool."""
        self.config = config or ConnectionPoolConfig()
        self._active_connections: Set[str] = set()
        self._idle_connections: asyncio.Queue = asyncio.Queue(maxsize=self.config.max_connections)
        self._connection_semaphore = asyncio.Semaphore(self.config.max_connections)
        self._stats = ResourcePoolStats()
        self._cleanup_task: Optional[asyncio.Task] = None
        self._lock = asyncio.Lock()
        
    async def start(self):
        """Start the connection pool and cleanup task."""
        logger.info(f"Starting async connection pool with {self.config.max_connections} max connections")
        
        # Pre-populate with minimum connections
        for _ in range(self.config.min_connections):
            connection_id = await self._create_connection()
            await self._idle_connections.put((connection_id, time.time()))
        
        # Start cleanup task
        self._cleanup_task = asyncio.create_task(self._cleanup_idle_connections())
        
    async def stop(self):
        """Stop the connection pool and cleanup resources."""
        logger.info("Stopping async connection pool")
        
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        # Clean up all connections
        while not self._idle_connections.empty():
            try:
                connection_id, _ = self._idle_connections.get_nowait()
                await self._close_connection(connection_id)
            except asyncio.QueueEmpty:
                break
    
    @asynccontextmanager
    async def get_connection(self) -> AsyncGenerator[str, None]:
        """
        Get a connection from the pool with proper resource management.
        
        Returns:
            Connection context manager for safe resource handling
        """
        async with self._connection_semaphore:
            self._stats.total_requests += 1
            start_time = time.time()
            
            try:
                # Try to get idle connection first
                connection_id = await self._get_idle_connection()
                if connection_id is None:
                    # Create new connection if needed
                    connection_id = await self._create_connection()
                
                async with self._lock:
                    self._active_connections.add(connection_id)
                    self._stats.active_connections = len(self._active_connections)
                
                try:
                    yield connection_id
                finally:
                    # Return connection to idle pool
                    await self._return_connection(connection_id)
                    
                    # Update stats
                    response_time = time.time() - start_time
                    self._update_response_time(response_time)
                    
            except Exception as e:
                self._stats.failed_requests += 1
                logger.error(f"Connection pool error: {e}")
                raise
    
    async def _get_idle_connection(self) -> Optional[str]:
        """Get connection from idle pool."""
        try:
            connection_id, created_time = await asyncio.wait_for(
                self._idle_connections.get(),
                timeout=0.1  # Quick timeout for non-blocking behavior
            )
            
            # Check if connection is still valid
            if time.time() - created_time < self.config.idle_timeout:
                return connection_id
            else:
                # Connection too old, close it
                await self._close_connection(connection_id)
                return None
                
        except asyncio.TimeoutError:
            return None
    
    async def _create_connection(self) -> str:
        """Create a new connection."""
        connection_id = f"dtesn_conn_{int(time.time() * 1000000)}"
        logger.debug(f"Created new connection: {connection_id}")
        return connection_id
    
    async def _close_connection(self, connection_id: str):
        """Close a connection and clean up resources."""
        logger.debug(f"Closing connection: {connection_id}")
        # In real implementation, would close actual connection resources
        
    async def _return_connection(self, connection_id: str):
        """Return connection to idle pool."""
        async with self._lock:
            if connection_id in self._active_connections:
                self._active_connections.remove(connection_id)
                self._stats.active_connections = len(self._active_connections)
        
        try:
            await asyncio.wait_for(
                self._idle_connections.put((connection_id, time.time())),
                timeout=0.1
            )
            async with self._lock:
                self._stats.idle_connections = self._idle_connections.qsize()
        except asyncio.TimeoutError:
            # Pool full, close connection
            await self._close_connection(connection_id)
    
    async def _cleanup_idle_connections(self):
        """Background task to clean up idle connections."""
        while True:
            try:
                await asyncio.sleep(60)  # Check every minute
                
                current_time = time.time()
                connections_to_remove = []
                
                # Check all idle connections
                temp_queue = asyncio.Queue()
                while not self._idle_connections.empty():
                    try:
                        connection_id, created_time = self._idle_connections.get_nowait()
                        if current_time - created_time > self.config.idle_timeout:
                            connections_to_remove.append(connection_id)
                        else:
                            await temp_queue.put((connection_id, created_time))
                    except asyncio.QueueEmpty:
                        break
                
                # Restore valid connections
                while not temp_queue.empty():
                    try:
                        item = temp_queue.get_nowait()
                        await self._idle_connections.put(item)
                    except asyncio.QueueEmpty:
                        break
                
                # Close expired connections
                for connection_id in connections_to_remove:
                    await self._close_connection(connection_id)
                    logger.debug(f"Cleaned up idle connection: {connection_id}")
                
                async with self._lock:
                    self._stats.idle_connections = self._idle_connections.qsize()
                    self._stats.pool_utilization = (
                        len(self._active_connections) / self.config.max_connections
                    )
                    self._stats.last_updated = current_time
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Connection cleanup error: {e}")
    
    def _update_response_time(self, response_time: float):
        """Update average response time with exponential moving average."""
        alpha = 0.1  # Smoothing factor
        if self._stats.avg_response_time == 0:
            self._stats.avg_response_time = response_time
        else:
            self._stats.avg_response_time = (
                alpha * response_time + (1 - alpha) * self._stats.avg_response_time
            )
    
    def get_stats(self) -> ResourcePoolStats:
        """Get current pool statistics."""
        return self._stats


class ConcurrencyManager:
    """
    Manages concurrent request processing with throttling and resource control.
    
    Provides request rate limiting, concurrent request management, and
    adaptive throttling based on system load.
    """
    
    def __init__(
        self,
        max_concurrent_requests: int = 50,
        max_requests_per_second: float = 100.0,
        burst_limit: int = 20
    ):
        """Initialize concurrency manager."""
        self.max_concurrent_requests = max_concurrent_requests
        self.max_requests_per_second = max_requests_per_second
        self.burst_limit = burst_limit
        
        self._request_semaphore = asyncio.Semaphore(max_concurrent_requests)
        self._rate_limiter = asyncio.Semaphore(burst_limit)
        self._request_times: List[float] = []
        self._lock = asyncio.Lock()
        
    @asynccontextmanager
    async def throttle_request(self) -> AsyncGenerator[None, None]:
        """
        Throttle request processing with rate limiting and concurrency control.
        
        Returns:
            Context manager for throttled request processing
        """
        # Apply rate limiting
        await self._apply_rate_limit()
        
        # Apply concurrency limiting
        async with self._request_semaphore:
            try:
                yield
            finally:
                # Clean up rate limiting state
                await self._cleanup_rate_limit()
    
    async def _apply_rate_limit(self):
        """Apply rate limiting based on requests per second."""
        current_time = time.time()
        
        async with self._lock:
            # Clean old request times (older than 1 second)
            self._request_times = [
                t for t in self._request_times if current_time - t < 1.0
            ]
            
            # Check if we're over the rate limit
            if len(self._request_times) >= self.max_requests_per_second:
                # Calculate delay needed
                oldest_time = min(self._request_times)
                delay = 1.0 - (current_time - oldest_time)
                if delay > 0:
                    await asyncio.sleep(delay)
            
            # Record this request time
            self._request_times.append(current_time)
    
    async def _cleanup_rate_limit(self):
        """Clean up rate limiting state after request completion."""
        current_time = time.time()
        
        # Remove old request timestamps beyond the rate limit window
        async with self._lock:
            # Clean up timestamps older than the rate limiting window
            self._request_times = [
                timestamp for timestamp in self._request_times
                if current_time - timestamp < 60.0
            ]
            
            # Log cleanup statistics for monitoring
            if len(self._request_times) > 0:
                logger.debug(
                    f"Rate limit cleanup: {len(self._request_times)} active timestamps remaining"
                )
    
    def get_current_load(self) -> Dict[str, Any]:
        """Get current concurrency and rate limiting statistics."""
        current_time = time.time()
        
        # Count recent requests
        recent_requests = len([
            t for t in self._request_times if current_time - t < 1.0
        ])
        
        return {
            "concurrent_requests": self.max_concurrent_requests - self._request_semaphore._value,
            "recent_requests_per_second": recent_requests,
            "rate_limit_utilization": recent_requests / self.max_requests_per_second,
            "concurrency_utilization": (
                (self.max_concurrent_requests - self._request_semaphore._value) 
                / self.max_concurrent_requests
            ),
            "available_slots": self._request_semaphore._value,
            "burst_capacity_remaining": self._rate_limiter._value
        }


class AsyncRequestQueue:
    """
    Enhanced async request queue with priority handling and load balancing.
    
    Provides non-blocking request queuing with priority levels, circuit breaker
    pattern, and adaptive load balancing based on system performance.
    """
    
    def __init__(
        self,
        max_queue_size: int = 1000,
        priority_levels: int = 3,
        circuit_breaker_threshold: int = 5,
        circuit_breaker_timeout: float = 60.0,
        adaptive_timeout: bool = True
    ):
        """Initialize async request queue with enhanced features."""
        self.max_queue_size = max_queue_size
        self.priority_levels = priority_levels
        self.circuit_breaker_threshold = circuit_breaker_threshold
        self.circuit_breaker_timeout = circuit_breaker_timeout
        self.adaptive_timeout = adaptive_timeout
        
        # Priority queues for different request types
        self._priority_queues = [
            asyncio.Queue(maxsize=max_queue_size // priority_levels)
            for _ in range(priority_levels)
        ]
        
        # Circuit breaker state
        self._circuit_breaker_failures = 0
        self._circuit_breaker_last_failure = 0.0
        self._circuit_breaker_open = False
        
        # Performance tracking for adaptive timeouts
        self._response_times = []
        self._success_rate = 1.0
        self._lock = asyncio.Lock()
        
        logger.info(f"AsyncRequestQueue initialized with {priority_levels} priority levels")
    
    async def enqueue_request(
        self,
        request_data: Any,
        priority: int = 1,
        timeout: Optional[float] = None
    ) -> str:
        """
        Enqueue request with priority and timeout handling.
        
        Args:
            request_data: Request data to process
            priority: Request priority (0=highest, 2=lowest)
            timeout: Optional custom timeout
            
        Returns:
            Request ID for tracking
        """
        # Check circuit breaker
        if self._circuit_breaker_open:
            current_time = time.time()
            if current_time - self._circuit_breaker_last_failure < self.circuit_breaker_timeout:
                raise RuntimeError("Circuit breaker is open - service temporarily unavailable")
            else:
                # Reset circuit breaker
                self._circuit_breaker_open = False
                self._circuit_breaker_failures = 0
                logger.info("Circuit breaker reset - service available")
        
        # Validate priority level
        priority = max(0, min(priority, self.priority_levels - 1))
        
        # Generate request ID
        request_id = f"req_{int(time.time() * 1000000)}_{priority}"
        
        # Calculate adaptive timeout if enabled
        if timeout is None and self.adaptive_timeout:
            timeout = self._calculate_adaptive_timeout()
        
        # Create request with metadata
        request_item = {
            "id": request_id,
            "data": request_data,
            "priority": priority,
            "timeout": timeout,
            "enqueued_at": time.time(),
            "retries": 0
        }
        
        try:
            # Add to appropriate priority queue (non-blocking)
            self._priority_queues[priority].put_nowait(request_item)
            logger.debug(f"Enqueued request {request_id} with priority {priority}")
            return request_id
            
        except asyncio.QueueFull:
            logger.warning(f"Queue full for priority {priority}, rejecting request {request_id}")
            raise RuntimeError(f"Request queue full for priority level {priority}")
    
    async def dequeue_request(self) -> Optional[Dict[str, Any]]:
        """
        Dequeue next request based on priority.
        
        Returns:
            Next request to process or None if no requests available
        """
        # Try queues in priority order (0=highest priority)
        for priority in range(self.priority_levels):
            try:
                # Non-blocking get
                request_item = self._priority_queues[priority].get_nowait()
                logger.debug(f"Dequeued request {request_item['id']} with priority {priority}")
                return request_item
            except asyncio.QueueEmpty:
                continue
        
        return None
    
    async def record_request_result(
        self,
        request_id: str,
        success: bool,
        response_time: float,
        error: Optional[str] = None
    ):
        """
        Record request completion for performance tracking and circuit breaker.
        
        Args:
            request_id: Request ID
            success: Whether request succeeded
            response_time: Request processing time
            error: Error message if failed
        """
        async with self._lock:
            # Update response time tracking
            self._response_times.append(response_time)
            if len(self._response_times) > 100:
                self._response_times = self._response_times[-100:]  # Keep last 100
            
            # Update success rate
            if success:
                self._success_rate = 0.95 * self._success_rate + 0.05 * 1.0
                # Reset circuit breaker failures on success
                if self._circuit_breaker_failures > 0:
                    self._circuit_breaker_failures = max(0, self._circuit_breaker_failures - 1)
            else:
                self._success_rate = 0.95 * self._success_rate + 0.05 * 0.0
                self._circuit_breaker_failures += 1
                self._circuit_breaker_last_failure = time.time()
                
                # Check if circuit breaker should open
                if self._circuit_breaker_failures >= self.circuit_breaker_threshold:
                    self._circuit_breaker_open = True
                    logger.warning(
                        f"Circuit breaker opened due to {self._circuit_breaker_failures} failures"
                    )
                
                logger.warning(f"Request {request_id} failed: {error}")
    
    def _calculate_adaptive_timeout(self) -> float:
        """Calculate adaptive timeout based on historical response times."""
        if not self._response_times:
            return 30.0  # Default timeout
        
        # Calculate 95th percentile response time
        sorted_times = sorted(self._response_times)
        p95_index = int(0.95 * len(sorted_times))
        p95_time = sorted_times[p95_index] if p95_index < len(sorted_times) else sorted_times[-1]
        
        # Add buffer based on success rate
        buffer_multiplier = 2.0 if self._success_rate < 0.9 else 1.5
        adaptive_timeout = p95_time * buffer_multiplier
        
        # Clamp between reasonable bounds
        return max(5.0, min(adaptive_timeout, 120.0))
    
    def get_queue_stats(self) -> Dict[str, Any]:
        """Get current queue statistics."""
        total_queued = sum(q.qsize() for q in self._priority_queues)
        queue_sizes = [q.qsize() for q in self._priority_queues]
        
        avg_response_time = (
            sum(self._response_times) / len(self._response_times)
            if self._response_times else 0.0
        )
        
        return {
            "total_queued_requests": total_queued,
            "priority_queue_sizes": queue_sizes,
            "queue_utilization": total_queued / self.max_queue_size,
            "circuit_breaker_open": self._circuit_breaker_open,
            "circuit_breaker_failures": self._circuit_breaker_failures,
            "success_rate": self._success_rate,
            "avg_response_time": avg_response_time,
            "adaptive_timeout": self._calculate_adaptive_timeout()
        }