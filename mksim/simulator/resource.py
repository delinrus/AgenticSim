"""
Resource management for fair-share allocation.

Resources (CPU, NPU, Memory, Network, Disk) have fixed capacity.
ResourceManager tracks capacity and provides helpers for allocation logic.
"""

from dataclasses import dataclass
from typing import Dict

from mksim.agentic.tool.tool_instance import ResourceType


@dataclass
class Resource:
    """
    System resource with fixed capacity.
    
    Attributes:
        resource_type: Type of resource
        total_capacity: Maximum capacity (work units per second)
    """
    resource_type: ResourceType
    total_capacity: float
    
    def __post_init__(self):
        """Validate capacity is positive."""
        if self.total_capacity <= 0:
            raise ValueError(
                f"Resource capacity must be positive, got {self.total_capacity} "
                f"for {self.resource_type.value}"
            )


class ResourceManager:
    """
    Manages system resources and their capacities.
    
    In fair-share model, resources are divided equally among active consumers.
    ResourceManager provides capacity lookup for scheduler calculations.
    """
    
    def __init__(self, resource_capacities: Dict[ResourceType, float]):
        """
        Initialize ResourceManager with resource capacities.
        
        Args:
            resource_capacities: Map from ResourceType to capacity value
            
        Example:
            resource_capacities = {
                ResourceType.CPU: 1000.0,      # 1000 CPU units/sec
                ResourceType.NPU: 256.0,       # 256 NPU units/sec
                ResourceType.MEMORY: 1024000.0,# 1GB/sec
                ResourceType.NETWORK: 10000.0, # 10GB/sec
                ResourceType.DISK: 5000.0      # 5GB/sec
            }
        """
        self.resources: Dict[ResourceType, Resource] = {}
        
        for resource_type, capacity in resource_capacities.items():
            self.resources[resource_type] = Resource(
                resource_type=resource_type,
                total_capacity=capacity
            )
    
    @classmethod
    def from_config(cls, config: Dict) -> 'ResourceManager':
        """
        Create ResourceManager from configuration dict.
        
        Args:
            config: Configuration with resource capacities
                   Expected format:
                   {
                       'cpu': {'capacity': 1000},
                       'npu': {'capacity': 256},
                       ...
                   }
        
        Returns:
            Initialized ResourceManager
        """
        resource_capacities = {}
        
        # Map config keys to ResourceType enum
        resource_map = {
            'cpu': ResourceType.CPU,
            'npu': ResourceType.NPU,
            'memory': ResourceType.MEMORY,
            'network': ResourceType.NETWORK,
            'disk': ResourceType.DISK,
        }
        
        for key, resource_type in resource_map.items():
            if key in config:
                capacity = config[key].get('capacity', 0.0)
                if capacity > 0:
                    resource_capacities[resource_type] = float(capacity)
        
        # Set defaults for missing resources (to avoid division by zero)
        for resource_type in ResourceType:
            if resource_type not in resource_capacities:
                # Default very high capacity (effectively unlimited)
                resource_capacities[resource_type] = 1e12
        
        return cls(resource_capacities)
    
    def get_capacity(self, resource_type: ResourceType) -> float:
        """
        Get total capacity for a resource type.
        
        Args:
            resource_type: Type of resource
            
        Returns:
            Total capacity (work units per second)
        """
        resource = self.resources.get(resource_type)
        if resource is None:
            # Return very high capacity for undefined resources
            return 1e12
        return resource.total_capacity
    
    def get_all_capacities(self) -> Dict[ResourceType, float]:
        """
        Get capacities for all resource types.
        
        Returns:
            Map from ResourceType to capacity
        """
        return {
            resource_type: resource.total_capacity
            for resource_type, resource in self.resources.items()
        }
    
    def __repr__(self) -> str:
        """String representation for debugging."""
        capacities = ", ".join(
            f"{rt.value}={cap:.0f}"
            for rt, cap in self.get_all_capacities().items()
        )
        return f"ResourceManager({capacities})"

