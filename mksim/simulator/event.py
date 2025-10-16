"""
Event system for discrete-event simulation.

Key design decision: EventQueue stores only START events (REQUEST_ARRIVAL, TOOL_START).
TOOL_FINISH events are computed dynamically at each simulation step.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any, List, Optional
from heapq import heappush, heappop


class EventType(Enum):
    """Types of events in the simulation."""
    REQUEST_ARRIVAL = 1  # New request arrives in the system
    TOOL_START = 2       # Tool instance starts execution


@dataclass(order=True)
class Event:
    """
    Discrete event in the simulation.
    
    Events are ordered by timestamp (primary) and priority (secondary).
    The payload is not compared for ordering.
    
    Attributes:
        timestamp: Simulation time when event occurs (seconds)
        event_type: Type of event
        payload: Event-specific data (not used for ordering)
        priority: Tiebreaker for events with same timestamp (lower = higher priority)
    """
    timestamp: float
    event_type: EventType = field(compare=False)
    priority: int = field(default=0, compare=True)
    payload: Dict[str, Any] = field(default_factory=dict, compare=False)
    
    def __post_init__(self):
        """Validate timestamp is non-negative."""
        if self.timestamp < 0:
            raise ValueError(f"Event timestamp must be non-negative, got {self.timestamp}")


class EventQueue:
    """
    Priority queue for simulation events, ordered by timestamp.
    
    Uses heapq for O(log n) push/pop operations.
    Only START events are stored in the queue.
    """
    
    def __init__(self):
        self._heap: List[Event] = []
        self._event_counter: int = 0  # For stable ordering of same-timestamp events
    
    def push(self, event: Event) -> None:
        """
        Add an event to the queue.
        
        Args:
            event: Event to schedule
        """
        heappush(self._heap, event)
        self._event_counter += 1
    
    def pop(self) -> Event:
        """
        Remove and return the next event (earliest timestamp).
        
        Returns:
            Next event to process
            
        Raises:
            IndexError: If queue is empty
        """
        if not self._heap:
            raise IndexError("Cannot pop from empty EventQueue")
        return heappop(self._heap)
    
    def peek(self) -> Optional[Event]:
        """
        Get the next event without removing it.
        
        Returns:
            Next event, or None if queue is empty
        """
        return self._heap[0] if self._heap else None
    
    def is_empty(self) -> bool:
        """Check if queue has no events."""
        return len(self._heap) == 0
    
    def size(self) -> int:
        """Get number of events in queue."""
        return len(self._heap)
    
    def clear(self) -> None:
        """Remove all events from queue."""
        self._heap.clear()
        self._event_counter = 0
    
    def __len__(self) -> int:
        """Support len() function."""
        return len(self._heap)
    
    def __bool__(self) -> bool:
        """Support bool() function - True if queue has events."""
        return bool(self._heap)

