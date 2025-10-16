"""
RequestGenerator - generates request arrivals following a Poisson process.
"""

import numpy as np
from typing import List, Dict, Optional
from uuid import uuid4

from mksim.simulator.event import Event, EventType
from mksim.agentic.request.request import Request
from utils.agentic_tool_graph import AgenticToolGraph


class RequestGenerator:
    """
    Generates request arrival events following a Poisson process.
    
    Supports multiple request types with different arrival rates and DAG templates.
    """
    
    def __init__(self, random_seed: Optional[int] = None):
        """
        Initialize request generator.
        
        Args:
            random_seed: Random seed for reproducibility (None for random)
        """
        self.random_seed = random_seed
        if random_seed is not None:
            np.random.seed(random_seed)
    
    def generate_poisson_arrivals(self,
                                  lambda_per_min: float,
                                  simulation_duration: float,
                                  start_time: float = 0.0) -> List[float]:
        """
        Generate arrival times following a Poisson process.
        
        Args:
            lambda_per_min: Average arrival rate (requests per minute)
            simulation_duration: Duration of simulation (seconds)
            start_time: Start time offset (seconds)
            
        Returns:
            List of arrival times (seconds)
        """
        lambda_per_sec = lambda_per_min / 60.0
        arrival_times = []
        current_time = start_time
        end_time = start_time + simulation_duration
        
        while current_time < end_time:
            # Inter-arrival time ~ Exponential(lambda)
            inter_arrival = np.random.exponential(1.0 / lambda_per_sec)
            current_time += inter_arrival
            
            if current_time < end_time:
                arrival_times.append(current_time)
        
        return arrival_times
    
    def generate_request_events(self,
                                request_type: str,
                                dag_template: AgenticToolGraph,
                                lambda_per_min: float,
                                simulation_duration: float,
                                start_time: float = 0.0) -> List[Event]:
        """
        Generate REQUEST_ARRIVAL events for a single request type.
        
        Args:
            request_type: Type/class of request
            dag_template: DAG template for this request type
            lambda_per_min: Arrival rate (requests per minute)
            simulation_duration: Duration of simulation (seconds)
            start_time: Start time offset (seconds)
            
        Returns:
            List of REQUEST_ARRIVAL events
        """
        arrival_times = self.generate_poisson_arrivals(
            lambda_per_min=lambda_per_min,
            simulation_duration=simulation_duration,
            start_time=start_time
        )
        
        events = []
        for arrival_time in arrival_times:
            # Create request from DAG template
            request = Request.create_from_dag(
                request_type=request_type,
                arrival_time=arrival_time,
                dag=dag_template,
                request_id=uuid4()
            )
            
            # Create arrival event
            event = Event(
                timestamp=arrival_time,
                event_type=EventType.REQUEST_ARRIVAL,
                payload={'request': request}
            )
            events.append(event)
        
        return events
    
    def generate_mixed_workload(self,
                                workload_specs: List[Dict],
                                simulation_duration: float,
                                start_time: float = 0.0) -> List[Event]:
        """
        Generate REQUEST_ARRIVAL events for multiple request types (mixed workload).
        
        Args:
            workload_specs: List of workload specifications, each containing:
                - 'request_type': str - type name
                - 'dag_template': AgenticToolGraph - DAG template
                - 'arrival_rate': float - lambda (requests per minute)
            simulation_duration: Duration of simulation (seconds)
            start_time: Start time offset (seconds)
            
        Returns:
            List of REQUEST_ARRIVAL events (sorted by timestamp)
            
        Example:
            workload_specs = [
                {
                    'request_type': 'web-search',
                    'dag_template': web_search_dag,
                    'arrival_rate': 60.0  # 60 req/min
                },
                {
                    'request_type': 'deep-research',
                    'dag_template': deep_research_dag,
                    'arrival_rate': 10.0  # 10 req/min
                }
            ]
        """
        all_events = []
        
        for spec in workload_specs:
            request_type = spec['request_type']
            dag_template = spec['dag_template']
            arrival_rate = spec['arrival_rate']
            
            events = self.generate_request_events(
                request_type=request_type,
                dag_template=dag_template,
                lambda_per_min=arrival_rate,
                simulation_duration=simulation_duration,
                start_time=start_time
            )
            all_events.extend(events)
        
        # Sort by timestamp (superposition of Poisson processes)
        all_events.sort(key=lambda e: e.timestamp)
        
        return all_events
    
    def generate_deterministic_arrivals(self,
                                       request_type: str,
                                       dag_template: AgenticToolGraph,
                                       num_requests: int,
                                       inter_arrival_time: float,
                                       start_time: float = 0.0) -> List[Event]:
        """
        Generate REQUEST_ARRIVAL events with deterministic (fixed) inter-arrival time.
        
        Useful for controlled experiments and validation.
        
        Args:
            request_type: Type/class of request
            dag_template: DAG template for this request type
            num_requests: Number of requests to generate
            inter_arrival_time: Fixed time between arrivals (seconds)
            start_time: Start time offset (seconds)
            
        Returns:
            List of REQUEST_ARRIVAL events
        """
        events = []
        
        for i in range(num_requests):
            arrival_time = start_time + i * inter_arrival_time
            
            # Create request from DAG template
            request = Request.create_from_dag(
                request_type=request_type,
                arrival_time=arrival_time,
                dag=dag_template,
                request_id=uuid4()
            )
            
            # Create arrival event
            event = Event(
                timestamp=arrival_time,
                event_type=EventType.REQUEST_ARRIVAL,
                payload={'request': request}
            )
            events.append(event)
        
        return events

