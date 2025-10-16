"""
Request - represents a user request with a DAG of tool executions.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional
from uuid import UUID, uuid4

from utils.agentic_tool_graph import AgenticToolGraph
from mksim.agentic.tool.tool_instance import ToolInstance


@dataclass
class Request:
    """
    User request that executes a DAG of tools.
    
    Attributes:
        request_id: Unique identifier for this request
        request_type: Type/class of request (e.g., "web-search", "deep-research")
        arrival_time: When request entered the system (simulation time)
        dag: Directed acyclic graph of tools to execute
        tool_instances: Map from tool_name to ToolInstance for this request
        start_time: When first tool started (None if not started)
        finish_time: When last tool finished (None if not finished)
    """
    request_id: UUID
    request_type: str
    arrival_time: float
    dag: AgenticToolGraph
    
    tool_instances: Dict[str, ToolInstance] = field(default_factory=dict)
    start_time: Optional[float] = field(default=None)
    finish_time: Optional[float] = field(default=None)
    
    @classmethod
    def create_from_dag(cls, 
                       request_type: str,
                       arrival_time: float,
                       dag: AgenticToolGraph,
                       request_id: Optional[UUID] = None) -> 'Request':
        """
        Create a Request with tool instances from a DAG template.
        
        Args:
            request_type: Type of request
            arrival_time: When request arrives
            dag: DAG template of tools
            request_id: Optional UUID (generated if not provided)
            
        Returns:
            New Request with initialized tool instances
        """
        if request_id is None:
            request_id = uuid4()
        
        request = cls(
            request_id=request_id,
            request_type=request_type,
            arrival_time=arrival_time,
            dag=dag
        )
        
        # Create tool instances for each node in DAG
        for tool_name in dag.nodes():
            tool_template = dag.get_task(tool_name)
            tool_id = f"{request_id}_{tool_name}"
            
            tool_instance = ToolInstance(
                tool_id=tool_id,
                tool_name=tool_name,
                request_id=request_id,
                tool_template=tool_template,
                request=request
            )
            request.tool_instances[tool_name] = tool_instance
        
        return request
    
    def is_completed(self) -> bool:
        """
        Check if all tools in the request have completed.
        
        Returns:
            True if all tool instances have status COMPLETED
        """
        from mksim.agentic.tool.tool_instance import ToolStatus
        return all(
            tool.status == ToolStatus.COMPLETED 
            for tool in self.tool_instances.values()
        )
    
    def get_latency(self) -> Optional[float]:
        """
        Calculate request latency (finish_time - arrival_time).
        
        Returns:
            Latency in seconds, or None if request not finished
        """
        if self.finish_time is None:
            return None
        return self.finish_time - self.arrival_time
    
    def get_root_tools(self) -> list[str]:
        """
        Get tool names that have no dependencies (entry points in DAG).
        
        Returns:
            List of tool names with in-degree 0
        """
        return [
            node for node in self.dag.graph.nodes
            if self.dag.graph.in_degree(node) == 0
        ]
    
    def get_dependencies(self, tool_name: str) -> list[str]:
        """
        Get names of tools that must complete before this tool can start.
        
        Args:
            tool_name: Name of tool in DAG
            
        Returns:
            List of dependency tool names
        """
        return list(self.dag.graph.predecessors(tool_name))
    
    def get_dependents(self, tool_name: str) -> list[str]:
        """
        Get names of tools that depend on this tool.
        
        Args:
            tool_name: Name of tool in DAG
            
        Returns:
            List of dependent tool names
        """
        return list(self.dag.graph.successors(tool_name))
    
    def can_start_tool(self, tool_name: str) -> bool:
        """
        Check if a tool can start (all dependencies completed).
        
        Args:
            tool_name: Name of tool to check
            
        Returns:
            True if all dependencies are completed
        """
        from mksim.agentic.tool.tool_instance import ToolStatus
        
        dependencies = self.get_dependencies(tool_name)
        return all(
            self.tool_instances[dep].status == ToolStatus.COMPLETED
            for dep in dependencies
        )
    
    def __repr__(self) -> str:
        """String representation for debugging."""
        status = "completed" if self.is_completed() else "in_progress"
        latency = self.get_latency()
        latency_str = f"{latency:.3f}s" if latency else "N/A"
        return (f"Request(id={self.request_id}, type={self.request_type}, "
                f"arrival={self.arrival_time:.3f}s, status={status}, "
                f"latency={latency_str})")

