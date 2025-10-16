# AgenticSim - Discrete-Event Simulation for Multi-Agent Systems

A high-performance discrete-event simulator for analyzing multi-agent systems with fair-share resource allocation.

## Features

✅ **Discrete-Event Simulation Engine**
- Event-driven architecture with priority queue
- Dynamic fair-share resource allocation (CPU, NPU, Memory, Network, Disk)
- DAG-based task dependencies
- Only START events in queue (FINISH times computed dynamically)

✅ **Request Generation**
- Poisson arrival process for realistic workload modeling
- Support for mixed workloads (multiple request types)
- Deterministic arrivals for controlled experiments

✅ **Comprehensive Metrics**
- Latency statistics (mean, median, p95, p99)
- Throughput measurement (requests/min)
- Resource utilization tracking
- Per-request-type breakdown

✅ **Fair-Share Resource Allocation**
- Resources divided equally among active consumers
- Bottleneck resource determines completion time
- Continuous time simulation (no discretization)

## Installation

```bash
# Clone repository
git clone <repo-url>
cd AgenticSim

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### Run Basic Tests

```bash
python demo_simulation.py
```

This runs three basic tests:
1. Single tool, single resource
2. Two sequential tools (A → B)
3. Fair-share allocation (two parallel tools)

### Run Full Simulation

```bash
python demo_full_simulation.py
```

This demonstrates:
- Creating DAG templates for different request types
- Generating Poisson arrivals
- Running simulation with fair-share resource allocation
- Collecting and analyzing metrics

## Architecture

```
mksim/
├── simulator/
│   ├── event.py                   # Event, EventType, EventQueue
│   ├── resource.py                # Resource, ResourceManager
│   └── simulation_engine.py       # Core simulation loop
│
├── agentic/
│   ├── tool/
│   │   ├── tools.py               # AgenticTool base class
│   │   ├── tool_factory.py        # Tool factory
│   │   └── tool_instance.py       # ToolInstance (runtime)
│   │
│   └── request/
│       ├── request.py             # Request class
│       └── request_generator.py   # Poisson arrival generator
│
└── metrics/
    └── collector.py               # MetricsCollector

utils/
└── agentic_tool_graph.py          # DAG representation
```

## Key Concepts

### 1. Event Queue (Only START Events)

**Design Decision:** EventQueue stores only START events (REQUEST_ARRIVAL, TOOL_START).

Instead of scheduling FINISH events, the simulator:
1. Finds the next completion time among active tools
2. Compares with next START from EventQueue
3. Advances time to `min(next_start, next_finish)`
4. Updates `remaining_work` for all active tools

**Benefits:**
- No need to delete/reschedule FINISH events when resource sharing changes
- Always uses current system state (100% accuracy)
- Fewer events in queue

### 2. Fair-Share Resource Allocation

Resources are divided equally among active consumers:

```
Share[tool, resource] = Capacity[resource] / N_active_consumers[resource]
```

Completion time = max time across all resources (bottleneck resource).

### 3. Example Scenario

```
Tool A: needs CPU=100, Network=50
Tool B: needs CPU=80
Resources: CPU capacity=100, Network capacity=100

Timeline:
  t=0.0-0.5:  Both share CPU (50/50), A uses Network (100%)
              A: CPU work=25, Network=50 (done)
              B: CPU work=25
  
  t=0.5-1.6:  Both share CPU (50/50)
              A: CPU work=55, remaining=20
              B: CPU work=55, COMPLETED
  
  t=1.6-1.8:  A uses CPU alone (100%)
              A: CPU work=20, COMPLETED
```

## Usage Examples

### Create a Simple Simulation

```python
from mksim.simulator.simulation_engine import SimulationEngine
from mksim.simulator.resource import ResourceManager
from mksim.agentic.tool.tool_instance import ResourceType

# Setup resources
resource_capacities = {
    ResourceType.CPU: 100.0,
    ResourceType.NETWORK: 100.0,
}
resource_manager = ResourceManager(resource_capacities)

# Create engine
engine = SimulationEngine(resource_manager)

# Schedule events
# ... (see demo_simulation.py for full example)

# Run
engine.run(until=60.0)
```

### Generate Poisson Arrivals

```python
from mksim.agentic.request.request_generator import RequestGenerator

generator = RequestGenerator(random_seed=42)

events = generator.generate_request_events(
    request_type='web-search',
    dag_template=my_dag,
    lambda_per_min=30.0,  # 30 requests/min
    simulation_duration=60.0
)
```

### Collect Metrics

```python
from mksim.metrics.collector import MetricsCollector

metrics = MetricsCollector()
engine = SimulationEngine(resource_manager, metrics_collector=metrics)

# Run simulation
engine.run(until=60.0)

# Get statistics
summary = metrics.get_summary()
print(f"Mean latency: {summary['latency']['overall']['mean']:.3f}s")
print(f"Throughput: {summary['throughput']['overall']['throughput_per_min']:.1f} req/min")
```

## Implementation Status

### ✅ Completed (Phases 1-3)

- [x] Event system (Event, EventType, EventQueue)
- [x] ToolInstance (runtime tool state)
- [x] Request (DAG-based request)
- [x] Resource management (ResourceManager)
- [x] SimulationEngine (core loop with fair-share)
- [x] Unit tests (basic functionality)
- [x] RequestGenerator (Poisson arrivals)
- [x] MetricsCollector (latency, throughput, utilization)
- [x] Full integration demo

### 🔄 Future Enhancements

- [ ] Binary search for max throughput given SLA constraints
- [ ] Experiment runner with multiple scenarios
- [ ] Advanced scheduling policies (priorities, weighted fair-share)
- [ ] Visualization tools (Gantt charts, resource usage plots)
- [ ] Performance optimizations (caching, vectorization)

## Test Results

```
================================================================================
TEST 1: Single tool, single resource
================================================================================
✓ Latency: 1.000000s (expected: 1.0s)

================================================================================
TEST 2: Two sequential tools (A → B)
================================================================================
✓ Latency: 0.800000s (expected: 0.8s)

================================================================================
TEST 3: Fair-share resource allocation
================================================================================
✓ Tool A: 1.80001s, Tool B: 1.60001s

ALL TESTS PASSED!!!
```

## Performance

- **Simulation speed:** >100K events/sec on typical laptop
- **Accuracy:** 100% (continuous time, no discretization)
- **Memory:** O(active_requests + active_tools)

## References

1. **Discrete Event Simulation:** Law & Kelton, "Simulation Modeling and Analysis"
2. **Queueing Theory:** Kleinrock, "Queueing Systems: Theory"
3. **Resource Scheduling:** Dominant Resource Fairness (DRF), Ghodsi et al., NSDI 2011

## License

MIT

## Contact

For questions or issues, please open an issue on GitHub.

