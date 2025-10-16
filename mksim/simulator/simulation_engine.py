"""
SimulationEngine - core discrete-event simulation loop.

Key design: EventQueue stores only START events.
FINISH times are computed dynamically at each step by finding the next resource completion.
"""

from collections import defaultdict
from typing import Dict, Set, Tuple, Optional
from uuid import UUID

from mksim.simulator.event import EventQueue, Event
from mksim.simulator.resource import ResourceManager
from mksim.agentic.request.request import Request
from mksim.agentic.tool.tool_instance import ToolInstance, ToolStatus, ResourceType

# Metrics collector is optional (imported dynamically to avoid circular dependency)
try:
    from mksim.metrics.collector import MetricsCollector
except ImportError:
    MetricsCollector = None


class SimulationEngine:
    """
    Discrete-event simulator with fair-share resource allocation.
    
    Architecture:
    - EventQueue contains only START events (REQUEST_ARRIVAL, TOOL_START)
    - At each step, computes next completion time among active tools
    - Advances time to min(next_start, next_completion)
    - Updates remaining_work for all active tools proportionally
    
    Attributes:
        resource_manager: Manages resource capacities
        current_time: Current simulation time (seconds)
        event_queue: Priority queue of START events
        requests: All requests in the system (by ID)
        active_tools: Currently running tool instances
    """
    
    def __init__(self, resource_manager: ResourceManager, metrics_collector=None):
        """
        Initialize simulation engine.
        
        Args:
            resource_manager: Resource capacity manager
            metrics_collector: Optional MetricsCollector for tracking metrics
        """
        self.resource_manager = resource_manager
        self.metrics = metrics_collector
        self.current_time: float = 0.0
        self.event_queue = EventQueue()
        
        # Request tracking
        self.requests: Dict[UUID, Request] = {}
        
        # Active tools (currently running)
        self.active_tools: Set[ToolInstance] = set()
        
        # Statistics
        self.completed_requests: list[Request] = []
        self.total_steps: int = 0
    
    def schedule_event(self, event: Event) -> None:
        """
        Add an event to the queue.
        
        Args:
            event: Event to schedule
        """
        self.event_queue.push(event)
    
    def run(self, until: float, max_steps: Optional[int] = None) -> None:
        """
        Run simulation until specified time or max steps.
        
        Args:
            until: Simulation end time (seconds)
            max_steps: Maximum number of steps (None for unlimited)
        """
        while True:
            # Check termination conditions
            if max_steps is not None and self.total_steps >= max_steps:
                break
            
            # Find next event time
            next_start_time = self.event_queue.peek().timestamp if self.event_queue else float('inf')
            
            # Check if simulation is done
            if next_start_time > until and not self.active_tools:
                break
            
            # Find next completion among active tools
            next_completion_time, completing_tool, completing_resource = self._find_next_completion()
            
            # Determine next time point
            next_time = min(next_start_time, next_completion_time)
            
            if next_time == float('inf'):
                # No more events
                break
            
            if next_time > until:
                # Simulation time limit reached
                break
            
            # Process event
            if next_time == next_start_time:
                # Handle tool start event
                event = self.event_queue.pop()
                self.current_time = event.timestamp
                self._handle_tool_start(event)
            else:
                # Handle resource completion(s)
                time_delta = next_completion_time - self.current_time
                self.current_time = next_completion_time
                self._handle_resource_completion(time_delta)
            
            # Take metrics snapshot if collector is available
            if self.metrics is not None:
                self.metrics.snapshot(self.current_time, self.active_tools, self.resource_manager)
            
            self.total_steps += 1
    
    def _update_resource_shares(self) -> None:
        """
        Update current_share for all active tools based on fair-share allocation.
        
        This method should be called whenever the set of active tools changes
        (tool starts or completes).
        """
        # Count consumers for each resource
        resource_consumers = defaultdict(list)
        for tool in self.active_tools:
            for resource_type in ResourceType:
                if tool.has_work_on_resource(resource_type):
                    resource_consumers[resource_type].append(tool)
        
        # Compute and assign fair share for each resource
        for resource_type in ResourceType:
            consumers = resource_consumers[resource_type]
            num_consumers = len(consumers)
            
            if num_consumers > 0:
                capacity = self.resource_manager.get_capacity(resource_type)
                share = capacity / num_consumers
                
                # Assign share to each consumer
                for tool in consumers:
                    tool.current_share[resource_type] = share
            
            # Set share to 0 for tools not consuming this resource
            non_consumers = self.active_tools - set(consumers)
            for tool in non_consumers:
                tool.current_share[resource_type] = 0.0
    
    def _find_next_completion(self) -> Tuple[float, Optional[ToolInstance], Optional[ResourceType]]:
        """
        Find the next resource completion among active tools.
        
        Returns:
            (completion_time, tool, resource_type) or (inf, None, None) if no active tools
        """
        if not self.active_tools:
            return float('inf'), None, None
        
        min_completion_time = float('inf')
        completing_tool = None
        completing_resource = None
        
        # For each tool and resource, compute completion time using pre-computed shares
        for tool in self.active_tools:
            for resource_type in ResourceType:
                if not tool.has_work_on_resource(resource_type):
                    continue
                
                remaining = tool.remaining_work[resource_type]
                # Use pre-computed fair share from tool instance
                share = tool.current_share[resource_type]
                
                if share <= 0:
                    continue
                
                # Time to complete work on this resource
                completion_time = self.current_time + (remaining / share)
                
                if completion_time < min_completion_time:
                    min_completion_time = completion_time
                    completing_tool = tool
                    completing_resource = resource_type
        
        return min_completion_time, completing_tool, completing_resource
    
    def _handle_tool_start(self, event: Event) -> None:
        """
        Handle tool start event.
        
        Registers request (if new), marks tool as RUNNING, initializes remaining_work.
        
        Args:
            event: Event with tool_instance
        """
        tool = event.tool_instance
        
        # Register request if this is the first time we see it
        if tool.request and tool.request.request_id not in self.requests:
            self.requests[tool.request.request_id] = tool.request
            tool.request.start_time = self.current_time
        
        # Update tool status and timing
        tool.status = ToolStatus.RUNNING
        tool.start_time = self.current_time
        
        # Initialize remaining work
        tool.initialize_work()
        
        # Add to active tools
        self.active_tools.add(tool)
        
        # Update resource shares for all active tools
        self._update_resource_shares()
    
    def _handle_resource_completion(self, time_delta: float) -> None:
        """
        Handle completion of work progress over a time delta and finalize
        all tools that completed at the current timestamp.
        
        Updates remaining_work for ALL active tools (time advanced by time_delta).
        Any tools that reach zero remaining work are marked COMPLETED and their
        dependents are considered for start.
        
        Args:
            time_delta: Time elapsed since last event
        """
        # Update each active tool's remaining work using their current share
        for active_tool in self.active_tools:
            for resource_type in ResourceType:
                if active_tool.has_work_on_resource(resource_type):
                    # Use pre-computed fair share from tool instance
                    share = active_tool.current_share[resource_type]
                    
                    # Work done = share × time_delta
                    work_done = share * time_delta
                    
                    # Update remaining (cannot be negative)
                    active_tool.remaining_work[resource_type] = max(
                        0.0, active_tool.remaining_work[resource_type] - work_done
                    )

        # Identify all tools that have completed at this timestamp
        finished_tools: list[ToolInstance] = [
            tool for tool in list(self.active_tools) if tool.is_completed()
        ]

        # Finalize all finished tools
        for tool in finished_tools:
            tool.status = ToolStatus.COMPLETED
            tool.finish_time = self.current_time
            # Safe removal after iteration
            if tool in self.active_tools:
                self.active_tools.remove(tool)
            # Check dependencies and start dependent tools
            self._check_and_start_dependents(tool)
        
        # Update resource shares after removing completed tools
        if finished_tools:
            self._update_resource_shares()
    
    def _check_and_start_dependents(self, finished_tool: ToolInstance) -> None:
        """
        Check if any tools can start now that finished_tool is done.
        
        Schedules START events for tools whose dependencies are all completed.
        Also checks if entire request is completed.
        
        Args:
            finished_tool: Tool that just completed
        """
        request = finished_tool.request
        if not request:
            return
        
        # Find dependent tools
        dependent_names = request.get_dependents(finished_tool.tool_name)
        
        for dependent_name in dependent_names:
            dependent_tool = request.tool_instances[dependent_name]
            
            # Check if all dependencies are completed
            if request.can_start_tool(dependent_name) and dependent_tool.status == ToolStatus.PENDING:
                # Schedule START event
                start_event = Event(
                    timestamp=self.current_time,
                    tool_instance=dependent_tool
                )
                self.schedule_event(start_event)
        
        # Check if entire request is completed
        if request.is_completed():
            request.finish_time = self.current_time
            self.completed_requests.append(request)
            
            # Record metrics if collector is available
            if self.metrics is not None:
                self.metrics.record_request_latency(request)
    
    
    def get_statistics(self) -> Dict:
        """
        Get simulation statistics.
        
        Returns:
            Dictionary with statistics
        """
        completed_count = len(self.completed_requests)
        total_requests = len(self.requests)
        
        latencies = [req.get_latency() for req in self.completed_requests if req.get_latency() is not None]
        
        return {
            'total_requests': total_requests,
            'completed_requests': completed_count,
            'active_tools': len(self.active_tools),
            'current_time': self.current_time,
            'total_steps': self.total_steps,
            'mean_latency': sum(latencies) / len(latencies) if latencies else 0.0,
            'min_latency': min(latencies) if latencies else 0.0,
            'max_latency': max(latencies) if latencies else 0.0,
        }

