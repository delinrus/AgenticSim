"""
Demo script to test basic simulation functionality.
"""

import networkx as nx
from mksim.simulator.simulation_engine import SimulationEngine
from mksim.simulator.resource import ResourceManager
from mksim.simulator.event import Event, EventType
from mksim.agentic.tool.tool_instance import ResourceType
from mksim.agentic.request.request import Request
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


def test_single_tool():
    """Test: Single tool using single resource."""
    print("\n" + "="*80)
    print("TEST 1: Single tool, single resource")
    print("="*80)
    
    # Setup resources
    resource_capacities = {
        ResourceType.CPU: 100.0,
        ResourceType.NPU: 100.0,
        ResourceType.MEMORY: 1000.0,
        ResourceType.NETWORK: 100.0,
        ResourceType.DISK: 100.0,
    }
    resource_manager = ResourceManager(resource_capacities)
    
    # Create a simple DAG with one tool
    graph = nx.DiGraph()
    tool = SimpleTool("test_tool", cpu_load=100.0)
    graph.add_node("test_tool", task=tool)
    
    dag = AgenticToolGraph(graph)
    
    # Create request
    request = Request.create_from_dag(
        request_type="test",
        arrival_time=0.0,
        dag=dag
    )
    
    # Create engine
    engine = SimulationEngine(resource_manager)
    
    # Schedule request arrival
    arrival_event = Event(
        timestamp=0.0,
        event_type=EventType.REQUEST_ARRIVAL,
        payload={'request': request}
    )
    engine.schedule_event(arrival_event)
    
    # Run simulation
    print(f"Running simulation...")
    engine.run(until=10.0)
    
    # Check results
    print(f"Request completed: {request.is_completed()}")
    print(f"Request latency: {request.get_latency():.6f} seconds")
    print(f"Expected latency: 1.0 seconds (100 work / 100 capacity)")
    print(f"Simulation steps: {engine.total_steps}")
    
    assert request.is_completed(), "Request should be completed"
    assert abs(request.get_latency() - 1.0) < 1e-6, f"Expected latency 1.0, got {request.get_latency()}"
    print("[PASS] Test passed!")


def test_sequential_tools():
    """Test: Two sequential tools."""
    print("\n" + "="*80)
    print("TEST 2: Two sequential tools (A -> B)")
    print("="*80)
    
    resource_capacities = {
        ResourceType.CPU: 100.0,
        ResourceType.NPU: 100.0,
        ResourceType.MEMORY: 1000.0,
        ResourceType.NETWORK: 100.0,
        ResourceType.DISK: 100.0,
    }
    resource_manager = ResourceManager(resource_capacities)
    
    # Create DAG: tool_a -> tool_b
    graph = nx.DiGraph()
    tool_a = SimpleTool("tool_a", cpu_load=50.0)
    tool_b = SimpleTool("tool_b", cpu_load=30.0)
    
    graph.add_node("tool_a", task=tool_a)
    graph.add_node("tool_b", task=tool_b)
    graph.add_edge("tool_a", "tool_b")
    
    dag = AgenticToolGraph(graph)
    
    # Create request
    request = Request.create_from_dag(
        request_type="test",
        arrival_time=0.0,
        dag=dag
    )
    
    # Create engine and run
    engine = SimulationEngine(resource_manager)
    arrival_event = Event(
        timestamp=0.0,
        event_type=EventType.REQUEST_ARRIVAL,
        payload={'request': request}
    )
    engine.schedule_event(arrival_event)
    
    print(f"Running simulation...")
    engine.run(until=10.0)
    
    # Check results
    print(f"Request completed: {request.is_completed()}")
    print(f"Request latency: {request.get_latency():.6f} seconds")
    print(f"Expected latency: 0.8 seconds (0.5 + 0.3)")
    print(f"Tool A finish time: {request.tool_instances['tool_a'].finish_time:.6f}s")
    print(f"Tool B finish time: {request.tool_instances['tool_b'].finish_time:.6f}s")
    print(f"Simulation steps: {engine.total_steps}")
    
    assert request.is_completed(), "Request should be completed"
    assert abs(request.get_latency() - 0.8) < 1e-6, f"Expected latency 0.8, got {request.get_latency()}"
    print("[PASS] Test passed!")


def test_fair_share():
    """Test: Two tools sharing CPU resource."""
    print("\n" + "="*80)
    print("TEST 3: Fair-share resource allocation (two parallel tools)")
    print("="*80)
    
    resource_capacities = {
        ResourceType.CPU: 100.0,
        ResourceType.NPU: 100.0,
        ResourceType.MEMORY: 1000.0,
        ResourceType.NETWORK: 100.0,
        ResourceType.DISK: 100.0,
    }
    resource_manager = ResourceManager(resource_capacities)
    
    # Create DAG with dummy root -> two parallel tools
    # This satisfies single-entry requirement while allowing parallel execution
    graph = nx.DiGraph()
    dummy_root = SimpleTool("dummy_root", cpu_load=0.001)  # Very small cost
    tool_a = SimpleTool("tool_a", cpu_load=100.0, network_load=50.0)
    tool_b = SimpleTool("tool_b", cpu_load=80.0)
    
    graph.add_node("dummy_root", task=dummy_root)
    graph.add_node("tool_a", task=tool_a)
    graph.add_node("tool_b", task=tool_b)
    graph.add_edge("dummy_root", "tool_a")
    graph.add_edge("dummy_root", "tool_b")
    
    dag = AgenticToolGraph(graph)
    
    # Create request
    request = Request.create_from_dag(
        request_type="test",
        arrival_time=0.0,
        dag=dag
    )
    
    # Create engine and run
    engine = SimulationEngine(resource_manager)
    arrival_event = Event(
        timestamp=0.0,
        event_type=EventType.REQUEST_ARRIVAL,
        payload={'request': request}
    )
    engine.schedule_event(arrival_event)
    
    print(f"Running simulation...")
    print(f"Tool A needs: CPU=100, Network=50")
    print(f"Tool B needs: CPU=80")
    print(f"CPU capacity: 100, Network capacity: 100")
    engine.run(until=10.0, max_steps=100)
    
    # Check results
    tool_a_instance = request.tool_instances["tool_a"]
    tool_b_instance = request.tool_instances["tool_b"]
    dummy_instance = request.tool_instances["dummy_root"]
    
    print(f"\nRequest completed: {request.is_completed()}")
    print(f"Dummy root finish time: {dummy_instance.finish_time}s")
    print(f"Tool A finish time: {tool_a_instance.finish_time}s (expected: ~1.8s)")
    print(f"Tool B finish time: {tool_b_instance.finish_time}s (expected: ~1.6s)")
    print(f"Request latency: {request.get_latency():.6f}s")
    print(f"Simulation steps: {engine.total_steps}")
    
    print(f"\nExplanation:")
    print(f"  t=0.0-0.5: Both tools share CPU (50/50), A uses Network (100%)")
    print(f"            A: CPU work done = 25, Network done = 50 (finished)")
    print(f"            B: CPU work done = 25")
    print(f"  t=0.5-1.6: Both tools share CPU (50/50)")
    print(f"            A: CPU work done = 55, remaining = 20")
    print(f"            B: CPU work done = 55, finished at t=1.6")
    print(f"  t=1.6-1.8: A uses CPU alone (100%)")
    print(f"            A: CPU work done = 20, finished at t=1.8")
    
    assert request.is_completed(), "Request should be completed"
    # Note: slight offset due to dummy_root (0.001 CPU load / 100 capacity = 0.00001s)
    assert abs(tool_b_instance.finish_time - 1.6) < 1e-4, f"Expected tool B finish at ~1.6, got {tool_b_instance.finish_time}"
    assert abs(tool_a_instance.finish_time - 1.8) < 1e-4, f"Expected tool A finish at ~1.8, got {tool_a_instance.finish_time}"
    print("\n[PASS] Test passed!")


if __name__ == '__main__':
    print("\n" + "="*80)
    print("DISCRETE-EVENT SIMULATION ENGINE - BASIC TESTS")
    print("="*80)
    
    try:
        test_single_tool()
        test_sequential_tools()
        test_fair_share()
        
        print("\n" + "="*80)
        print("ALL TESTS PASSED!!!")
        print("="*80)
        
    except AssertionError as e:
        print(f"\n[FAIL] Test failed: {e}")
        raise
    except Exception as e:
        print(f"\n[ERROR] Error: {e}")
        import traceback
        traceback.print_exc()
        raise

