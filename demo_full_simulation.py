"""
Full simulation demo with request generation and metrics collection.

This demonstrates:
1. Creating DAG templates for different request types
2. Generating Poisson arrivals
3. Running simulation with fair-share resource allocation
4. Collecting and analyzing metrics
"""

import networkx as nx
from mksim.simulator.simulation_engine import SimulationEngine
from mksim.simulator.resource import ResourceManager
from mksim.agentic.tool.tool_instance import ResourceType
from mksim.agentic.request.request_generator import RequestGenerator
from mksim.metrics.collector import MetricsCollector
from mksim.agentic.tool.tools import AgenticTool
from utils.agentic_tool_graph import AgenticToolGraph


class SimpleTool(AgenticTool):
    """Simple tool for testing."""
    
    def __init__(self, name, cpu_load=0.0, network_load=0.0, npu_load=0.0, memory_load=0.0, disk_load=0.0):
        super().__init__(name, {})
        self._cpu_load = cpu_load
        self._network_load = network_load
        self._npu_load = npu_load
        self._memory_load = memory_load
        self._disk_load = disk_load
    
    def get_cpu_load(self):
        return self._cpu_load
    
    def get_network_load(self):
        return self._network_load
    
    def get_npu_load(self):
        return self._npu_load
    
    def get_memory_load(self):
        return self._memory_load
    
    def get_disk_load(self):
        return self._disk_load


def create_web_search_dag() -> AgenticToolGraph:
    """
    Create DAG for web-search request type.
    
    Flow: query_planning -> web_search -> result_ranking
    """
    graph = nx.DiGraph()
    
    # Tools
    query_planning = SimpleTool("query_planning", cpu_load=10.0)
    web_search = SimpleTool("web_search", network_load=50.0, cpu_load=5.0)
    result_ranking = SimpleTool("result_ranking", cpu_load=20.0)
    
    # Nodes
    graph.add_node("query_planning", task=query_planning)
    graph.add_node("web_search", task=web_search)
    graph.add_node("result_ranking", task=result_ranking)
    
    # Edges
    graph.add_edge("query_planning", "web_search")
    graph.add_edge("web_search", "result_ranking")
    
    return AgenticToolGraph(graph)


def create_deep_research_dag() -> AgenticToolGraph:
    """
    Create DAG for deep-research request type.
    
    Flow: research_planning -> [literature_search, data_analysis] -> synthesis
    """
    graph = nx.DiGraph()
    
    # Tools
    research_planning = SimpleTool("research_planning", cpu_load=15.0)
    literature_search = SimpleTool("literature_search", network_load=100.0, cpu_load=10.0)
    data_analysis = SimpleTool("data_analysis", cpu_load=80.0, memory_load=500.0)
    synthesis = SimpleTool("synthesis", cpu_load=40.0)
    
    # Nodes
    graph.add_node("research_planning", task=research_planning)
    graph.add_node("literature_search", task=literature_search)
    graph.add_node("data_analysis", task=data_analysis)
    graph.add_node("synthesis", task=synthesis)
    
    # Edges
    graph.add_edge("research_planning", "literature_search")
    graph.add_edge("research_planning", "data_analysis")
    graph.add_edge("literature_search", "synthesis")
    graph.add_edge("data_analysis", "synthesis")
    
    return AgenticToolGraph(graph)


def main():
    """Run full simulation demo."""
    print("\n" + "="*80)
    print("FULL DISCRETE-EVENT SIMULATION DEMO")
    print("="*80)
    
    # Setup resources
    print("\n1. Setting up system resources...")
    resource_capacities = {
        ResourceType.CPU: 100.0,        # 100 CPU units/sec
        ResourceType.NPU: 100.0,        # 100 NPU units/sec
        ResourceType.MEMORY: 1000.0,    # 1000 memory units/sec
        ResourceType.NETWORK: 100.0,    # 100 network units/sec
        ResourceType.DISK: 100.0,       # 100 disk units/sec
    }
    resource_manager = ResourceManager(resource_capacities)
    print(f"   {resource_manager}")
    
    # Create DAG templates
    print("\n2. Creating DAG templates for request types...")
    web_search_dag = create_web_search_dag()
    deep_research_dag = create_deep_research_dag()
    print(f"   - Web Search DAG: {len(web_search_dag.nodes())} tools")
    print(f"   - Deep Research DAG: {len(deep_research_dag.nodes())} tools")
    
    # Setup metrics collector
    print("\n3. Initializing metrics collector...")
    metrics = MetricsCollector()
    
    # Setup simulation engine
    print("\n4. Creating simulation engine...")
    engine = SimulationEngine(resource_manager)
    # Inject metrics collector into engine
    engine.metrics = metrics
    
    # Generate requests
    print("\n5. Generating request arrivals (Poisson process)...")
    generator = RequestGenerator(random_seed=42)
    
    simulation_duration = 60.0  # 60 seconds
    
    workload_specs = [
        {
            'request_type': 'web-search',
            'dag_template': web_search_dag,
            'arrival_rate': 30.0  # 30 requests/min
        },
        {
            'request_type': 'deep-research',
            'dag_template': deep_research_dag,
            'arrival_rate': 10.0  # 10 requests/min
        }
    ]
    
    arrival_events = generator.generate_mixed_workload(
        workload_specs=workload_specs,
        simulation_duration=simulation_duration
    )
    
    print(f"   Generated {len(arrival_events)} tool start events")
    
    # Count unique requests by type
    unique_requests = {}
    for e in arrival_events:
        req = e.tool_instance.request
        if req.request_id not in unique_requests:
            unique_requests[req.request_id] = req.request_type
    
    web_search_count = sum(1 for rt in unique_requests.values() if rt == 'web-search')
    deep_research_count = sum(1 for rt in unique_requests.values() if rt == 'deep-research')
    
    print(f"   Total unique requests: {len(unique_requests)}")
    print(f"   - Web Search: {web_search_count} requests")
    print(f"   - Deep Research: {deep_research_count} requests")
    
    # Schedule all arrival events
    for event in arrival_events:
        engine.schedule_event(event)
    
    # Run simulation
    print("\n6. Running simulation...")
    print(f"   Duration: {simulation_duration} seconds")
    
    metrics.simulation_start_time = 0.0
    engine.run(until=simulation_duration*2)
    metrics.simulation_end_time = engine.current_time
    
    print(f"   Simulation completed!")
    print(f"   - Total steps: {engine.total_steps}")
    print(f"   - Final time: {engine.current_time:.2f}s")
    
    # Collect metrics from completed requests
    print("\n7. Collecting metrics...")
    for request in engine.completed_requests:
        metrics.record_request_latency(request)
    
    # Print summary
    print("\n8. Results:")
    metrics.print_summary()
    
    # Additional analysis
    print("\n" + "="*80)
    print("SIMULATION SUMMARY")
    print("="*80)
    print(f"Total requests generated: {len(arrival_events)}")
    print(f"Total requests completed: {len(engine.completed_requests)}")
    print(f"Completion rate: {len(engine.completed_requests)/len(arrival_events)*100:.1f}%")
    print(f"Active tools at end: {len(engine.active_tools)}")
    print(f"Total simulation steps: {engine.total_steps}")
    print("="*80)


if __name__ == '__main__':
    main()

