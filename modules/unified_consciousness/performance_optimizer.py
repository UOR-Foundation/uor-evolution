"""
Performance Optimization Module for Unified Consciousness Framework

This module provides performance monitoring, optimization, and resource management
for the unified consciousness system.
"""

import asyncio
import time
import psutil
import gc
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
from functools import wraps
import numpy as np
from collections import deque
import weakref

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Performance metrics for consciousness operations"""
    operation_name: str
    execution_time_ms: float
    memory_usage_mb: float
    cpu_usage_percent: float
    timestamp: datetime = field(default_factory=datetime.now)
    success: bool = True
    error: Optional[str] = None


@dataclass
class ResourceLimits:
    """Resource limits for consciousness system"""
    max_memory_mb: float = 1000.0
    max_cpu_percent: float = 50.0
    max_concurrent_operations: int = 4
    operation_timeout_seconds: float = 30.0
    cache_size_mb: float = 100.0


@dataclass
class OptimizationProfile:
    """Optimization profile for different operation modes"""
    name: str
    memory_aggressive: bool = False
    cpu_aggressive: bool = False
    cache_enabled: bool = True
    batch_size: int = 10
    parallel_execution: bool = True


class PerformanceOptimizer:
    """
    Performance optimizer for the unified consciousness framework
    """
    
    def __init__(self, resource_limits: Optional[ResourceLimits] = None):
        """Initialize performance optimizer"""
        self.resource_limits = resource_limits or ResourceLimits()
        self.metrics_history = deque(maxlen=1000)
        self.operation_cache = weakref.WeakValueDictionary()
        self.active_operations = set()
        self.optimization_profile = OptimizationProfile(name="balanced")
        
        # Performance monitoring
        self.monitoring_enabled = True
        self.performance_alerts = []
        
        # Resource tracking
        self.process = psutil.Process()
        self.baseline_memory = self.process.memory_info().rss / 1024 / 1024
        
        logger.info("Performance Optimizer initialized")
    
    def performance_monitor(self, operation_name: str):
        """
        Decorator to monitor performance of async operations
        """
        def decorator(func: Callable):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                if not self.monitoring_enabled:
                    return await func(*args, **kwargs)
                
                # Check resource availability
                self._monitor_operation(operation_name)
                
                # Track operation
                operation_id = f"{operation_name}_{time.time()}"
                self.active_operations.add(operation_id)
                
                # Start monitoring
                start_time = time.time()
                start_memory = self.process.memory_info().rss / 1024 / 1024
                start_cpu = self.process.cpu_percent(interval=0.1)
                
                try:
                    # Execute operation with timeout
                    result = await asyncio.wait_for(
                        func(*args, **kwargs),
                        timeout=self.resource_limits.operation_timeout_seconds
                    )
                    
                    # Record metrics
                    end_time = time.time()
                    end_memory = self.process.memory_info().rss / 1024 / 1024
                    end_cpu = self.process.cpu_percent(interval=0.1)
                    
                    metrics = PerformanceMetrics(
                        operation_name=operation_name,
                        execution_time_ms=(end_time - start_time) * 1000,
                        memory_usage_mb=end_memory - start_memory,
                        cpu_usage_percent=(start_cpu + end_cpu) / 2,
                        success=True
                    )
                    
                    self._record_metrics(metrics)
                    return result
                    
                except asyncio.TimeoutError:
                    metrics = PerformanceMetrics(
                        operation_name=operation_name,
                        execution_time_ms=self.resource_limits.operation_timeout_seconds * 1000,
                        memory_usage_mb=0,
                        cpu_usage_percent=0,
                        success=False,
                        error="Operation timeout"
                    )
                    self._record_metrics(metrics)
                    raise
                    
                except Exception as e:
                    end_time = time.time()
                    metrics = PerformanceMetrics(
                        operation_name=operation_name,
                        execution_time_ms=(end_time - start_time) * 1000,
                        memory_usage_mb=0,
                        cpu_usage_percent=0,
                        success=False,
                        error=str(e)
                    )
                    self._record_metrics(metrics)
                    raise
                    
                finally:
                    self.active_operations.discard(operation_id)
                    
            return wrapper
        return decorator
    
    def cache_result(self, cache_key: str, ttl_seconds: float = 300):
        """
        Decorator to cache operation results
        """
        def decorator(func: Callable):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                if not self.optimization_profile.cache_enabled:
                    return await func(*args, **kwargs)
                
                # Generate cache key
                full_key = f"{cache_key}_{str(args)}_{str(kwargs)}"
                
                # Check cache
                if full_key in self.operation_cache:
                    cached_result = self.operation_cache[full_key]
                    if hasattr(cached_result, '_cache_time'):
                        if time.time() - cached_result._cache_time < ttl_seconds:
                            return cached_result
                
                # Execute and cache
                result = await func(*args, **kwargs)
                result._cache_time = time.time()
                
                # Check cache size
                if self._get_cache_size() < self.resource_limits.cache_size_mb:
                    self.operation_cache[full_key] = result
                
                return result
                
            return wrapper
        return decorator
    
    async def batch_operations(
        self,
        operations: List[Callable],
        batch_size: Optional[int] = None
    ) -> List[Any]:
        """
        Execute operations in optimized batches
        """
        batch_size = batch_size or self.optimization_profile.batch_size
        results = []
        
        for i in range(0, len(operations), batch_size):
            batch = operations[i:i + batch_size]
            
            if self.optimization_profile.parallel_execution:
                # Parallel execution
                batch_results = await asyncio.gather(*[op() for op in batch])
            else:
                # Sequential execution
                batch_results = []
                for op in batch:
                    result = await op()
                    batch_results.append(result)
            
            results.extend(batch_results)
            
            # Allow other operations
            await asyncio.sleep(0)
        
        return results
    
    def optimize_memory(self):
        """
        Perform memory optimization
        """
        if self.optimization_profile.memory_aggressive:
            # Force garbage collection
            gc.collect()
            
            # Clear caches
            self.operation_cache.clear()
            
            # Trim metrics history
            if len(self.metrics_history) > 500:
                self.metrics_history = deque(
                    list(self.metrics_history)[-500:],
                    maxlen=1000
                )
        
        current_memory = self.process.memory_info().rss / 1024 / 1024
        memory_freed = current_memory - self.process.memory_info().rss / 1024 / 1024
        
        logger.info(f"Memory optimization freed {memory_freed:.2f} MB")
        return memory_freed
    
    def set_optimization_profile(self, profile: OptimizationProfile):
        """
        Set optimization profile
        """
        self.optimization_profile = profile
        logger.info(f"Optimization profile set to: {profile.name}")
        
        # Apply profile settings
        if profile.memory_aggressive:
            self.optimize_memory()
        
        if profile.cpu_aggressive:
            # Reduce parallel operations
            self.resource_limits.max_concurrent_operations = 2
    
    def get_performance_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive performance report
        """
        if not self.metrics_history:
            return {
                'status': 'no_data',
                'message': 'No performance data collected yet'
            }
        
        # Analyze metrics
        metrics_by_operation = {}
        for metric in self.metrics_history:
            if metric.operation_name not in metrics_by_operation:
                metrics_by_operation[metric.operation_name] = []
            metrics_by_operation[metric.operation_name].append(metric)
        
        # Calculate statistics
        report = {
            'summary': {
                'total_operations': len(self.metrics_history),
                'success_rate': sum(1 for m in self.metrics_history if m.success) / len(self.metrics_history),
                'active_operations': len(self.active_operations),
                'cache_size_mb': self._get_cache_size(),
                'current_memory_mb': self.process.memory_info().rss / 1024 / 1024,
                'current_cpu_percent': self.process.cpu_percent(interval=0.1)
            },
            'operations': {}
        }
        
        for op_name, metrics in metrics_by_operation.items():
            execution_times = [m.execution_time_ms for m in metrics if m.success]
            memory_usage = [m.memory_usage_mb for m in metrics if m.success]
            cpu_usage = [m.cpu_usage_percent for m in metrics if m.success]
            
            report['operations'][op_name] = {
                'count': len(metrics),
                'success_rate': sum(1 for m in metrics if m.success) / len(metrics),
                'avg_execution_time_ms': np.mean(execution_times) if execution_times else 0,
                'max_execution_time_ms': np.max(execution_times) if execution_times else 0,
                'avg_memory_usage_mb': np.mean(memory_usage) if memory_usage else 0,
                'avg_cpu_usage_percent': np.mean(cpu_usage) if cpu_usage else 0
            }
        
        # Add alerts
        report['alerts'] = self.performance_alerts
        
        return report
    
    def check_performance_thresholds(self) -> List[Dict[str, Any]]:
        """
        Check if performance is within acceptable thresholds
        """
        alerts = []
        
        # Check memory usage
        current_memory = self.process.memory_info().rss / 1024 / 1024
        if current_memory > self.resource_limits.max_memory_mb:
            alerts.append({
                'type': 'memory_exceeded',
                'severity': 'high',
                'message': f'Memory usage ({current_memory:.2f} MB) exceeds limit ({self.resource_limits.max_memory_mb} MB)',
                'timestamp': datetime.now()
            })
        
        # Check CPU usage
        cpu_percent = self.process.cpu_percent(interval=0.1)
        if cpu_percent > self.resource_limits.max_cpu_percent:
            alerts.append({
                'type': 'cpu_exceeded',
                'severity': 'medium',
                'message': f'CPU usage ({cpu_percent:.1f}%) exceeds limit ({self.resource_limits.max_cpu_percent}%)',
                'timestamp': datetime.now()
            })
        
        # Check operation latency
        recent_metrics = list(self.metrics_history)[-100:]
        if recent_metrics:
            slow_operations = [
                m for m in recent_metrics
                if m.execution_time_ms > 1000 and m.success
            ]
            if len(slow_operations) > 10:
                alerts.append({
                    'type': 'high_latency',
                    'severity': 'medium',
                    'message': f'{len(slow_operations)} operations exceeded 1000ms in recent history',
                    'timestamp': datetime.now()
                })
        
        self.performance_alerts = alerts
        return alerts
    
    async def auto_optimize(self):
        """
        Automatically optimize based on current performance
        """
        alerts = self.check_performance_thresholds()
        
        for alert in alerts:
            if alert['type'] == 'memory_exceeded':
                # Switch to memory-aggressive profile
                memory_profile = OptimizationProfile(
                    name="memory_saver",
                    memory_aggressive=True,
                    cache_enabled=False,
                    batch_size=5
                )
                self.set_optimization_profile(memory_profile)
                
            elif alert['type'] == 'cpu_exceeded':
                # Switch to CPU-conservative profile
                cpu_profile = OptimizationProfile(
                    name="cpu_saver",
                    cpu_aggressive=True,
                    parallel_execution=False,
                    batch_size=3
                )
                self.set_optimization_profile(cpu_profile)
                
            elif alert['type'] == 'high_latency':
                # Optimize for latency
                latency_profile = OptimizationProfile(
                    name="low_latency",
                    cache_enabled=True,
                    parallel_execution=True,
                    batch_size=20
                )
                self.set_optimization_profile(latency_profile)
        
        # If no alerts, use balanced profile
        if not alerts and self.optimization_profile.name != "balanced":
            balanced_profile = OptimizationProfile(name="balanced")
            self.set_optimization_profile(balanced_profile)
    
    # Private methods
    
    def _check_resource_availability(self) -> bool:
        """Check if resources are available for new operation"""
        # Check concurrent operations
        if len(self.active_operations) >= self.resource_limits.max_concurrent_operations:
            return False
        
        # Check memory
        current_memory = self.process.memory_info().rss / 1024 / 1024
        if current_memory > self.resource_limits.max_memory_mb * 0.9:
            return False
        
        # Check CPU
        cpu_percent = self.process.cpu_percent(interval=0.1)
        if cpu_percent > self.resource_limits.max_cpu_percent * 0.9:
            return False
        
        return True
    
    def _record_metrics(self, metrics: PerformanceMetrics):
        """Record performance metrics"""
        self.metrics_history.append(metrics)
        
        # Log slow operations
        if metrics.execution_time_ms > 500:
            logger.warning(
                f"Slow operation: {metrics.operation_name} "
                f"took {metrics.execution_time_ms:.2f}ms"
            )
        
        # Log failed operations
        if not metrics.success:
            logger.error(
                f"Operation failed: {metrics.operation_name} - {metrics.error}"
            )
    
    def _get_cache_size(self) -> float:
        """Estimate cache size in MB"""
        # Simplified estimation
        return len(self.operation_cache) * 0.1  # Assume 0.1 MB per cached item

    def _monitor_operation(self, operation_name: str) -> None:
        """Ensure resources are available, raising ResourceError if not."""
        if not self._check_resource_availability():
            memory_mb = self.process.memory_info().rss / 1024 / 1024
            cpu_percent = self.process.cpu_percent(interval=0.1)
            raise ResourceError(
                "Resource limits exceeded",
                operation_name=operation_name,
                memory_mb=memory_mb,
                cpu_percent=cpu_percent,
            )


class ResourceError(Exception):
    """Raised when resource limits are exceeded"""

    def __init__(self, message: str, operation_name: str, memory_mb: float, cpu_percent: float) -> None:
        super().__init__(message)
        self.operation_name = operation_name
        self.memory_mb = memory_mb
        self.cpu_percent = cpu_percent

    def __str__(self) -> str:  # pragma: no cover - simple formatting
        base = super().__str__()
        return (
            f"{base} (operation={self.operation_name}, "
            f"memory={self.memory_mb:.2f} MB, cpu={self.cpu_percent:.1f}%)"
        )


# Predefined optimization profiles
OPTIMIZATION_PROFILES = {
    'balanced': OptimizationProfile(
        name='balanced',
        memory_aggressive=False,
        cpu_aggressive=False,
        cache_enabled=True,
        batch_size=10,
        parallel_execution=True
    ),
    'memory_saver': OptimizationProfile(
        name='memory_saver',
        memory_aggressive=True,
        cpu_aggressive=False,
        cache_enabled=False,
        batch_size=5,
        parallel_execution=True
    ),
    'cpu_saver': OptimizationProfile(
        name='cpu_saver',
        memory_aggressive=False,
        cpu_aggressive=True,
        cache_enabled=True,
        batch_size=3,
        parallel_execution=False
    ),
    'low_latency': OptimizationProfile(
        name='low_latency',
        memory_aggressive=False,
        cpu_aggressive=False,
        cache_enabled=True,
        batch_size=20,
        parallel_execution=True
    ),
    'high_throughput': OptimizationProfile(
        name='high_throughput',
        memory_aggressive=False,
        cpu_aggressive=False,
        cache_enabled=True,
        batch_size=50,
        parallel_execution=True
    )
}


# Performance monitoring utilities

async def measure_operation_performance(
    operation: Callable,
    iterations: int = 10
) -> Dict[str, float]:
    """
    Measure average performance of an operation
    """
    execution_times = []
    memory_usage = []
    
    process = psutil.Process()
    
    for _ in range(iterations):
        start_time = time.time()
        start_memory = process.memory_info().rss / 1024 / 1024
        
        await operation()
        
        end_time = time.time()
        end_memory = process.memory_info().rss / 1024 / 1024
        
        execution_times.append((end_time - start_time) * 1000)
        memory_usage.append(end_memory - start_memory)
        
        # Small delay between iterations
        await asyncio.sleep(0.1)
    
    return {
        'avg_execution_time_ms': np.mean(execution_times),
        'std_execution_time_ms': np.std(execution_times),
        'min_execution_time_ms': np.min(execution_times),
        'max_execution_time_ms': np.max(execution_times),
        'avg_memory_usage_mb': np.mean(memory_usage),
        'total_iterations': iterations
    }


def profile_consciousness_operation(func: Callable):
    """
    Decorator to profile consciousness operations
    """
    @wraps(func)
    async def wrapper(*args, **kwargs):
        import cProfile
        import pstats
        from io import StringIO
        
        profiler = cProfile.Profile()
        profiler.enable()
        
        try:
            result = await func(*args, **kwargs)
        finally:
            profiler.disable()
            
            # Generate profile report
            stream = StringIO()
            stats = pstats.Stats(profiler, stream=stream)
            stats.sort_stats('cumulative')
            stats.print_stats(20)  # Top 20 functions
            
            logger.debug(f"Profile for {func.__name__}:\n{stream.getvalue()}")
        
        return result
    
    return wrapper
