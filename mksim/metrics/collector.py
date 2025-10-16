"""
MetricsCollector - collects and computes simulation metrics.
"""

from collections import defaultdict
from typing import Dict, List, Set, Optional
import numpy as np

from mksim.agentic.request.request import Request
from mksim.agentic.tool.tool_instance import ToolInstance, ResourceType
from mksim.simulator.resource import ResourceManager


class MetricsCollector:
    """
    Collects metrics during simulation and computes statistics.
    
    Tracks:
    - Request latencies (per type and overall)
    - Throughput (completed requests per time)
    - Resource utilization over time
    - Queue depths and active tool counts
    """
    
    def __init__(self):
        """Initialize metrics collector."""
        # Request latencies by type
        self.latencies_by_type: Dict[str, List[float]] = defaultdict(list)
        
        # All completed requests (for detailed analysis)
        self.completed_requests: List[Request] = []
        
        # Resource utilization snapshots
        self.utilization_snapshots: List[Dict] = []
        
        # Active tools count over time
        self.active_tools_snapshots: List[Dict] = []
        
        # Simulation start/end times
        self.simulation_start_time: Optional[float] = None
        self.simulation_end_time: Optional[float] = None
    
    def record_request_latency(self, request: Request) -> None:
        """
        Record latency for a completed request.
        
        Args:
            request: Completed request
        """
        latency = request.get_latency()
        if latency is not None:
            self.latencies_by_type[request.request_type].append(latency)
            self.completed_requests.append(request)
    
    def snapshot(self, 
                 current_time: float,
                 active_tools: Set[ToolInstance],
                 resource_manager: ResourceManager) -> None:
        """
        Take a snapshot of current system state.
        
        Called periodically during simulation to track resource utilization.
        
        Args:
            current_time: Current simulation time
            active_tools: Set of currently running tools
            resource_manager: Resource manager
        """
        # Compute actual resource consumption and utilization
        utilization = {}
        resource_consumers = defaultdict(int)
        
        # Calculate utilization using pre-computed shares from tool instances
        for resource_type in ResourceType:
            capacity = resource_manager.get_capacity(resource_type)
            
            if capacity == 0:
                utilization[resource_type.value] = 0.0
                continue
            
            # Sum actual consumption from all active tools
            # Each tool's consumption is stored in tool.current_share[resource_type]
            total_consumption = sum(
                tool.current_share[resource_type]
                for tool in active_tools
                if tool.has_work_on_resource(resource_type)
            )
            
            # Count consumers for tracking
            num_consumers = sum(
                1 for tool in active_tools
                if tool.has_work_on_resource(resource_type)
            )
            resource_consumers[resource_type] = num_consumers
            
            utilization[resource_type.value] = total_consumption / capacity
        
        self.utilization_snapshots.append({
            'time': current_time,
            'utilization': utilization,
            'consumers': dict(resource_consumers)
        })
        
        # Track active tools count
        self.active_tools_snapshots.append({
            'time': current_time,
            'count': len(active_tools)
        })
    
    def get_latency_statistics(self, request_type: Optional[str] = None) -> Dict:
        """
        Get latency statistics for a request type or all requests.
        
        Args:
            request_type: Type of request (None for all)
            
        Returns:
            Dictionary with statistics (mean, median, p50, p95, p99, min, max)
        """
        if request_type is None:
            # Aggregate all latencies
            latencies = []
            for type_latencies in self.latencies_by_type.values():
                latencies.extend(type_latencies)
        else:
            latencies = self.latencies_by_type.get(request_type, [])
        
        if not latencies:
            return {
                'count': 0,
                'mean': 0.0,
                'median': 0.0,
                'p50': 0.0,
                'p95': 0.0,
                'p99': 0.0,
                'min': 0.0,
                'max': 0.0,
                'std': 0.0
            }
        
        latencies_array = np.array(latencies)
        
        return {
            'count': len(latencies),
            'mean': float(np.mean(latencies_array)),
            'median': float(np.median(latencies_array)),
            'p50': float(np.percentile(latencies_array, 50)),
            'p95': float(np.percentile(latencies_array, 95)),
            'p99': float(np.percentile(latencies_array, 99)),
            'min': float(np.min(latencies_array)),
            'max': float(np.max(latencies_array)),
            'std': float(np.std(latencies_array))
        }
    
    def get_throughput(self, request_type: Optional[str] = None) -> Dict:
        """
        Get throughput statistics.
        
        Args:
            request_type: Type of request (None for all)
            
        Returns:
            Dictionary with throughput metrics
        """
        if request_type is None:
            total_requests = sum(len(l) for l in self.latencies_by_type.values())
        else:
            total_requests = len(self.latencies_by_type.get(request_type, []))
        
        # Compute simulation duration
        if self.simulation_end_time is not None and self.simulation_start_time is not None:
            duration = self.simulation_end_time - self.simulation_start_time
        else:
            # Use snapshots as fallback
            if self.utilization_snapshots:
                duration = self.utilization_snapshots[-1]['time'] - self.utilization_snapshots[0]['time']
            else:
                duration = 1.0  # Avoid division by zero
        
        if duration <= 0:
            duration = 1.0
        
        throughput_per_sec = total_requests / duration
        throughput_per_min = throughput_per_sec * 60.0
        
        return {
            'total_requests': total_requests,
            'duration': duration,
            'throughput_per_sec': throughput_per_sec,
            'throughput_per_min': throughput_per_min
        }
    
    def get_resource_utilization(self) -> Dict:
        """
        Get average resource utilization over simulation.
        
        Returns:
            Dictionary with average utilization per resource
        """
        if not self.utilization_snapshots:
            return {}
        
        # Time-weighted average utilization
        avg_utilization = {}
        
        for resource_type in ResourceType:
            resource_key = resource_type.value
            
            # Compute time-weighted average
            total_weighted_util = 0.0
            total_time = 0.0
            
            for i in range(len(self.utilization_snapshots) - 1):
                snapshot = self.utilization_snapshots[i]
                next_snapshot = self.utilization_snapshots[i + 1]
                
                util = snapshot['utilization'].get(resource_key, 0.0)
                time_delta = next_snapshot['time'] - snapshot['time']
                
                total_weighted_util += util * time_delta
                total_time += time_delta
            
            if total_time > 0:
                avg_utilization[resource_key] = total_weighted_util / total_time
            else:
                avg_utilization[resource_key] = 0.0
        
        return avg_utilization
    
    def get_summary(self) -> Dict:
        """
        Get comprehensive summary of all metrics.
        
        Returns:
            Dictionary with all metrics
        """
        summary = {
            'latency': {},
            'throughput': {},
            'utilization': {}
        }
        
        # Overall latency statistics
        summary['latency']['overall'] = self.get_latency_statistics()
        
        # Per-type latency statistics
        for request_type in self.latencies_by_type.keys():
            summary['latency'][request_type] = self.get_latency_statistics(request_type)
        
        # Overall throughput
        summary['throughput']['overall'] = self.get_throughput()
        
        # Per-type throughput
        for request_type in self.latencies_by_type.keys():
            summary['throughput'][request_type] = self.get_throughput(request_type)
        
        # Resource utilization
        summary['utilization'] = self.get_resource_utilization()
        
        return summary
    
    def print_summary(self) -> None:
        """Print a formatted summary of metrics."""
        summary = self.get_summary()
        
        print("\n" + "="*80)
        print("SIMULATION METRICS SUMMARY")
        print("="*80)
        
        # Latency
        print("\nLATENCY STATISTICS:")
        print("-" * 80)
        for req_type, stats in summary['latency'].items():
            if stats['count'] > 0:
                print(f"\n{req_type.upper()}:")
                print(f"  Count: {stats['count']}")
                print(f"  Mean:   {stats['mean']:.6f}s")
                print(f"  Median: {stats['median']:.6f}s")
                print(f"  P95:    {stats['p95']:.6f}s")
                print(f"  P99:    {stats['p99']:.6f}s")
                print(f"  Min:    {stats['min']:.6f}s")
                print(f"  Max:    {stats['max']:.6f}s")
        
        # Throughput
        print("\nTHROUGHPUT:")
        print("-" * 80)
        for req_type, stats in summary['throughput'].items():
            print(f"\n{req_type.upper()}:")
            print(f"  Total requests: {stats['total_requests']}")
            print(f"  Duration: {stats['duration']:.2f}s")
            print(f"  Throughput: {stats['throughput_per_min']:.2f} req/min")
        
        # Utilization
        print("\nRESOURCE UTILIZATION:")
        print("-" * 80)
        for resource, util in summary['utilization'].items():
            print(f"  {resource.upper():10s}: {util*100:.1f}%")
        
        print("\n" + "="*80)

