"""
ToolInstance - runtime instance of a tool for a specific request.

Each request creates its own instances of tools from the DAG.
ToolInstance tracks execution state and remaining work per resource.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional
from uuid import UUID

from mksim.agentic.tool.tools import AgenticTool
from mksim.common.constants import EPSILON


class ToolStatus(Enum):
    """Execution status of a tool instance."""
    PENDING = "pending"      # Not started yet, waiting for dependencies
    RUNNING = "running"      # Currently executing
    COMPLETED = "completed"  # Finished execution


class ResourceType(Enum):
    """Types of system resources."""
    CPU = "cpu"
    NPU = "npu"
    MEMORY = "memory"
    NETWORK = "network"
    DISK = "disk"


@dataclass
class ToolInstance:
    """
    Runtime instance of a tool for a specific request.
    
    Attributes:
        tool_id: Unique identifier (format: "{request_id}_{tool_name}")
        tool_name: Name of the tool (node name in DAG)
        request_id: ID of the parent request
        tool_template: Reference to the tool definition (AgenticTool)
        status: Current execution status
        start_time: When tool started executing (None if not started)
        finish_time: When tool completed (None if not finished)
        remaining_work: Work remaining for each resource type
    """
    tool_id: str
    tool_name: str
    request_id: UUID
    tool_template: AgenticTool
    
    status: ToolStatus = field(default=ToolStatus.PENDING)
    start_time: Optional[float] = field(default=None)
    finish_time: Optional[float] = field(default=None)
    
    # Remaining work per resource (initialized when tool starts)
    remaining_work: Dict[ResourceType, float] = field(default_factory=dict, compare=False, hash=False)
    
    def __hash__(self):
        """Make ToolInstance hashable based on tool_id."""
        return hash(self.tool_id)
    
    def __eq__(self, other):
        """Check equality based on tool_id."""
        if not isinstance(other, ToolInstance):
            return False
        return self.tool_id == other.tool_id
    
    def __post_init__(self):
        """Initialize remaining_work to zero for all resources."""
        if not self.remaining_work:
            self.remaining_work = {resource_type: 0.0 for resource_type in ResourceType}
    
    def initialize_work(self) -> None:
        """
        Initialize remaining_work from tool template when tool starts.
        Called when status changes to RUNNING.
        """
        self.remaining_work = {
            ResourceType.CPU: self.tool_template.get_cpu_load(),
            ResourceType.NPU: self.tool_template.get_npu_load(),
            ResourceType.MEMORY: self.tool_template.get_memory_load(),
            ResourceType.NETWORK: self.tool_template.get_network_load(),
            ResourceType.DISK: self.tool_template.get_disk_load(),
        }
    
    def get_load(self, resource_type: ResourceType) -> float:
        """
        Get the total work needed for a resource type.
        
        Args:
            resource_type: Type of resource
            
        Returns:
            Total work units needed (from tool template)
        """
        if resource_type == ResourceType.CPU:
            return self.tool_template.get_cpu_load()
        elif resource_type == ResourceType.NPU:
            return self.tool_template.get_npu_load()
        elif resource_type == ResourceType.MEMORY:
            return self.tool_template.get_memory_load()
        elif resource_type == ResourceType.NETWORK:
            return self.tool_template.get_network_load()
        elif resource_type == ResourceType.DISK:
            return self.tool_template.get_disk_load()
        else:
            return 0.0
    
    def is_completed(self, epsilon: float = EPSILON) -> bool:
        """
        Check if tool has completed all work.
        
        Args:
            epsilon: Tolerance for floating-point comparison
            
        Returns:
            True if all remaining_work values are near zero
        """
        return all(work <= epsilon for work in self.remaining_work.values())
    
    def has_work_on_resource(self, resource_type: ResourceType, epsilon: float = EPSILON) -> bool:
        """
        Check if tool still has work remaining on a resource.
        
        Args:
            resource_type: Type of resource
            epsilon: Tolerance for floating-point comparison
            
        Returns:
            True if remaining work > epsilon
        """
        return self.remaining_work.get(resource_type, 0.0) > epsilon
    
    def __repr__(self) -> str:
        """String representation for debugging."""
        return (f"ToolInstance(id={self.tool_id}, name={self.tool_name}, "
                f"status={self.status.value}, start={self.start_time}, "
                f"finish={self.finish_time})")

