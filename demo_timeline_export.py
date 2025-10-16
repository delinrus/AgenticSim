"""
Demo: Export resource allocation timeline for visualization.

This script demonstrates how to export resource allocation timeline
which can be used for creating Gantt-like charts showing resource usage.
"""

import json
import networkx as nx
from mksim.simulator.simulation_engine import SimulationEngine
from mksim.simulator.resource import ResourceManager
from mksim.agentic.request.request import Request
from mksim.simulator.event import Event
from mksim.metrics.collector import MetricsCollector
from utils.agentic_tool_graph import AgenticToolGraph
from mksim.agentic.tool.tools import AgenticTool


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


def create_simple_dag():
    """Create a simple DAG with parallel and sequential tools."""
    graph = nx.DiGraph()
    
    # Dummy root (very small load)
    dummy_root = SimpleTool(name="start", cpu_load=0.001)
    
    # Tool A: CPU and Network
    tool_a = SimpleTool(
        name="extract_text",
        cpu_load=50.0,
        network_load=30.0
    )
    
    # Tool B: CPU only (parallel with A)
    tool_b = SimpleTool(
        name="fetch_data",
        cpu_load=40.0
    )
    
    # Tool C: CPU (sequential after A and B)
    tool_c = SimpleTool(
        name="process_results",
        cpu_load=60.0
    )
    
    # Build graph
    graph.add_node("start", task=dummy_root)
    graph.add_node("extract_text", task=tool_a)
    graph.add_node("fetch_data", task=tool_b)
    graph.add_node("process_results", task=tool_c)
    
    # Edges
    graph.add_edge("start", "extract_text")
    graph.add_edge("start", "fetch_data")
    graph.add_edge("extract_text", "process_results")
    graph.add_edge("fetch_data", "process_results")
    
    return AgenticToolGraph(graph)


def main():
    print("="*80)
    print("RESOURCE ALLOCATION TIMELINE EXPORT DEMO")
    print("="*80)
    print()
    
    # Setup resources
    print("1. Setting up resources...")
    from mksim.agentic.tool.tool_instance import ResourceType
    resource_capacities = {
        ResourceType.CPU: 100,
        ResourceType.MEMORY: 1000,
        ResourceType.NETWORK: 100
    }
    resource_manager = ResourceManager(resource_capacities)
    print(f"   {resource_manager}")
    print()
    
    # Create DAG
    print("2. Creating DAG...")
    dag = create_simple_dag()
    print(f"   DAG created with parallel and sequential tools")
    print()
    
    # Create requests
    print("3. Creating requests...")
    requests = []
    for i in range(3):
        request = Request.create_from_dag(
            request_type="test",
            arrival_time=float(i),
            dag=dag
        )
        requests.append(request)
    print(f"   Created {len(requests)} requests")
    print()
    
    # Setup metrics collector
    print("4. Creating metrics collector...")
    metrics = MetricsCollector()
    print("   Metrics collector initialized")
    print()
    
    # Create simulation engine
    print("5. Creating simulation engine...")
    engine = SimulationEngine(resource_manager, metrics_collector=metrics)
    print("   Engine ready")
    print()
    
    # Schedule events
    print("6. Scheduling events...")
    for request in requests:
        root_tools = request.get_root_tools()
        for tool_name in root_tools:
            tool_instance = request.tool_instances[tool_name]
            event = Event(
                timestamp=request.arrival_time,
                tool_instance=tool_instance
            )
            engine.schedule_event(event)
    print(f"   Scheduled events for {len(requests)} requests")
    print()
    
    # Run simulation
    print("7. Running simulation...")
    engine.run(until=10.0)
    print(f"   Simulation completed!")
    print(f"   - Total steps: {engine.total_steps}")
    print(f"   - Final time: {engine.current_time:.2f}s")
    print()
    
    # Export timeline
    print("8. Exporting resource allocation timeline...")
    timeline_data = metrics.export_resource_timeline()
    print(f"   Timeline exported for {len(timeline_data['resources'])} resources")
    print()
    
    # Show timeline summary
    print("9. Timeline Summary:")
    print("-"*80)
    for resource in timeline_data['resources']:
        resource_type = resource['type']
        timeline = resource['timeline']
        
        if not timeline:
            print(f"\n{resource_type.upper()}: No activity")
            continue
        
        print(f"\n{resource_type.upper()}:")
        print(f"  Total intervals: {len(timeline)}")
        
        # Show first few intervals
        num_to_show = min(5, len(timeline))
        print(f"  First {num_to_show} intervals:")
        
        for i, interval in enumerate(timeline[:num_to_show]):
            start = interval['start']
            end = interval['end']
            total = interval['total']
            allocations = interval['allocations']
            
            print(f"    [{i+1}] t={start:.3f}-{end:.3f}s (duration={end-start:.3f}s)")
            print(f"        Total allocated: {total:.1f}")
            
            if allocations:
                print(f"        Tools:")
                for tool_id, share in allocations.items():
                    print(f"          - {tool_id}: {share:.1f}")
            else:
                print(f"        Tools: (none)")
    
    print()
    
    # Save to JSON file
    output_file = "resource_timeline.json"
    print(f"10. Saving timeline to {output_file}...")
    with open(output_file, 'w') as f:
        json.dump(timeline_data, f, indent=2)
    print(f"    Timeline saved!")
    print()
    
    print("="*80)
    print("DEMO COMPLETED!")
    print("="*80)
    print()
    print(f"Timeline data saved to: {output_file}")
    print("You can use this data to create visualizations like Gantt charts.")
    print()


if __name__ == "__main__":
    main()

