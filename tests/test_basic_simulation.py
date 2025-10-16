"""
Basic integration tests for simulation engine.
"""

import unittest
from uuid import uuid4

from mksim.simulator.simulation_engine import SimulationEngine
from mksim.simulator.resource import ResourceManager
from mksim.simulator.event import Event, EventType
from mksim.agentic.tool.tool_instance import ResourceType
from mksim.agentic.request.request import Request
from mksim.agentic.tool.tools import AgenticTool
from utils.agentic_tool_graph import AgenticToolGraph
import networkx as nx


class SimpleTool(AgenticTool):
    """Simple tool for testing."""
    
    def __init__(self, name, cpu_load=0, network_load=0, npu_load=0, memory_load=0, disk_load=0):
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


class TestBasicSimulation(unittest.TestCase):
    """Test basic simulation functionality."""
    
    def setUp(self):
        """Set up test resources."""
        self.resource_capacities = {
            ResourceType.CPU: 100.0,
            ResourceType.NPU: 100.0,
            ResourceType.MEMORY: 1000.0,
            ResourceType.NETWORK: 100.0,
            ResourceType.DISK: 100.0,
        }
        self.resource_manager = ResourceManager(self.resource_capacities)
    
    def test_single_tool_single_resource(self):
        """Test: Single tool using single resource should complete in work/capacity time."""
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
        engine = SimulationEngine(self.resource_manager)
        
        # Schedule request arrival
        arrival_event = Event(
            timestamp=0.0,
            event_type=EventType.REQUEST_ARRIVAL,
            payload={'request': request}
        )
        engine.schedule_event(arrival_event)
        
        # Run simulation
        engine.run(until=10.0)
        
        # Check results
        self.assertTrue(request.is_completed())
        latency = request.get_latency()
        
        # Expected latency: 100 work / 100 capacity = 1.0 second
        self.assertAlmostEqual(latency, 1.0, places=6)
    
    def test_two_tools_sequential(self):
        """Test: Two sequential tools (A -> B) should complete in sum of their times."""
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
        engine = SimulationEngine(self.resource_manager)
        arrival_event = Event(
            timestamp=0.0,
            event_type=EventType.REQUEST_ARRIVAL,
            payload={'request': request}
        )
        engine.schedule_event(arrival_event)
        engine.run(until=10.0)
        
        # Check results
        self.assertTrue(request.is_completed())
        latency = request.get_latency()
        
        # Expected: tool_a (50/100 = 0.5s) + tool_b (30/100 = 0.3s) = 0.8s
        self.assertAlmostEqual(latency, 0.8, places=6)
    
    def test_fair_share_two_tools(self):
        """Test: Two tools running simultaneously should share resources fairly."""
        # Create two independent tools (both start immediately)
        graph = nx.DiGraph()
        tool_a = SimpleTool("tool_a", cpu_load=100.0, network_load=50.0)
        tool_b = SimpleTool("tool_b", cpu_load=80.0)
        
        graph.add_node("tool_a", task=tool_a)
        graph.add_node("tool_b", task=tool_b)
        # No edges - both are root tools
        
        dag = AgenticToolGraph(graph)
        
        # Create request
        request = Request.create_from_dag(
            request_type="test",
            arrival_time=0.0,
            dag=dag
        )
        
        # Create engine and run
        engine = SimulationEngine(self.resource_manager)
        arrival_event = Event(
            timestamp=0.0,
            event_type=EventType.REQUEST_ARRIVAL,
            payload={'request': request}
        )
        engine.schedule_event(arrival_event)
        engine.run(until=10.0)
        
        # Check results
        self.assertTrue(request.is_completed())
        
        # tool_a needs: CPU=100, Network=50
        # tool_b needs: CPU=80
        # 
        # Both start at t=0
        # Network: tool_a gets 100% = 100/s → finishes at t=0.5s
        # CPU: both share 50/50 = 50/s each
        #   - tool_a: at t=0.5, did 50*0.5=25 CPU work, remaining=75
        #   - tool_b: at t=0.5, did 50*0.5=25 CPU work, remaining=55
        # After t=0.5, tool_a only uses CPU
        #   - tool_a: needs 75 more, shares with tool_b → 50/s each
        #   - tool_b: needs 55 more
        #   - tool_b finishes at: 0.5 + 55/50 = 0.5 + 1.1 = 1.6s
        # After t=1.6, tool_a alone
        #   - remaining for tool_a: 75 - 50*1.1 = 75 - 55 = 20
        #   - tool_a finishes at: 1.6 + 20/100 = 1.8s
        
        tool_a_instance = request.tool_instances["tool_a"]
        tool_b_instance = request.tool_instances["tool_b"]
        
        self.assertAlmostEqual(tool_b_instance.finish_time, 1.6, places=6)
        self.assertAlmostEqual(tool_a_instance.finish_time, 1.8, places=6)
        self.assertAlmostEqual(request.get_latency(), 1.8, places=6)


if __name__ == '__main__':
    unittest.main()

