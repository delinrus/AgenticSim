from __future__ import annotations

from typing import Iterable, List, Tuple, Dict, Any

import networkx as nx
from matplotlib import pyplot as plt

from mksim.agentic.tool.tool_factory import ToolFactory
from mksim.agentic.tool.tools import AgenticTool


class AgenticToolGraph:
    """
    Build a DAG of tasks from config and provide:
      - cycle detection
      - single-entry & reachability validation
      - sequential execution (one-after-another)
      - total latency (sum in topo order)
      - simple drawing helper
    """

    def __init__(self, graph: nx.DiGraph):
        self.graph = graph
        self._detect_cycles_and_fail()
        self._validate_single_entry_and_reachability()

    # ----------------------------
    # Graph construction
    # ----------------------------

    @classmethod
    def build_from_config(cls, cfg: Dict[str, Any]) -> AgenticToolGraph:
        graph: nx.DiGraph = nx.DiGraph()
        agents = cfg.get("Agents", {})
        if not isinstance(agents, dict):
            raise TypeError("cfg['Agents'] must be a mapping of stages to their configs")

        # 1) Add nodes and bind Task objects
        for stage_name, stage_conf in agents.items():
            tools = (stage_conf or {}).get("tools", {})
            if not isinstance(tools, dict):
                raise TypeError(f"Agents['{stage_name}']['tools'] must be a mapping of task_name -> task_params")

            for tool_name, tool_params in tools.items():
                task_obj = ToolFactory.create_tool(
                    tool_name=tool_name,
                    tool_type = tool_params.get("task_type"),
                    config=tool_params
                )
                graph.add_node(
                    tool_name,
                    task=task_obj,
                    stage=stage_name,
                    task_type=tool_params.get("task_type"),
                    config=tool_params,
                )

        # 2) Add directed edges: dependency -> task
        for stage_name, stage_conf in agents.items():
            tools = (stage_conf or {}).get("tools", {})
            for tool_name, tool_params in tools.items():
                for dep in tool_params.get("dependencies", []) or []:
                    if dep not in graph:
                        raise KeyError(f"Dependency '{dep}' for task '{tool_name}' is not defined as a node")
                    graph.add_edge(dep, tool_name)

        return cls(graph)


    @classmethod
    def build_from_objects(cls, tools: list[AgenticTool], dependency_map: dict[str, list[str]]) -> AgenticToolGraph:
        graph: nx.DiGraph = nx.DiGraph()

        for tool in tools:
            graph.add_node(
                tool.name,
                task=tool
            )

        for tool_name, dependencies in dependency_map.items():
            for dependency in dependencies:
                if dependency not in graph:
                    raise KeyError(f"Dependency '{dependency}' for tool '{tool_name}' is not defined as a node")
                graph.add_edge(dependency, tool_name)

        return cls(graph)

    # ----------------------------
    # Validation
    # ----------------------------
    def _detect_cycles_and_fail(self) -> None:
        if not nx.is_directed_acyclic_graph(self.graph):
            cycle = next(nx.simple_cycles(self.graph))
            path = " -> ".join(cycle + [cycle[0]])
            raise ValueError(f"Cycle detected in task graph: {path}")

    def _validate_single_entry_and_reachability(self) -> None:
        """
        Ensure:
          1) Exactly one entry node (in-degree == 0).
          2) Every node is reachable from that entry node (following edge direction).
        """
        entry_nodes = [n for n, deg in self.graph.in_degree() if deg == 0]

        if len(entry_nodes) == 0:
            raise ValueError("No entry node found (no node with in-degree == 0).")
        if len(entry_nodes) > 1:
            details = ", ".join(map(str, entry_nodes))
            raise ValueError(f"Multiple entry nodes found (expected exactly one): {details}")

        entry = entry_nodes[0]

        # Reachability (directed): descendants plus the entry itself
        reachable = set(nx.descendants(self.graph, entry)) | {entry}
        if len(reachable) != self.graph.number_of_nodes():
            unreachable = set(self.graph.nodes) - reachable
            details = ", ".join(map(str, unreachable))
            raise ValueError(
                f"Unreachable nodes from entry '{entry}': {details}. "
                "All nodes must be reachable from the single starting point."
            )

    # ----------------------------
    # Execution & latency
    # ----------------------------
    def topological_order(self) -> Iterable[str]:
        """Return a valid topological ordering of the DAG."""
        # (Redundant guards are cheap and make this safe to call standalone.)
        self._detect_cycles_and_fail()
        self._validate_single_entry_and_reachability()
        return nx.topological_sort(self.graph)

    # ----------------------------
    # Utilities
    # ----------------------------
    def get_task(self, node_name: str) -> AgenticTool:
        return self.graph.nodes[node_name]["task"]

    def nodes(self) -> List[str]:
        return list(self.graph.nodes)

    def edges(self) -> List[Tuple[str, str]]:
        return list(self.graph.edges)

    def get_mandatory_nodes(self):
        G = self.graph
        start = [n for n in G.nodes if G.in_degree(n) == 0][0]
        end = [n for n in G.nodes if G.out_degree(n) == 0][0]

        all_paths = list(nx.all_simple_paths(G, start, end))

        common_nodes = set(all_paths[0])
        for path in all_paths[1:]:
            common_nodes &= set(path)

        topo_sorted = list(nx.topological_sort(G))
        return [n for n in topo_sorted if n in common_nodes]

    # ----------------------------
    # Visualization
    # ----------------------------
    def draw(self) -> None:
        pos = nx.spring_layout(self.graph, seed=42)
        plt.figure(figsize=(8, 6))

        nx.draw_networkx_nodes(
            self.graph, pos,
            node_size=900, node_color="#9ecae1", edgecolors="#225ea8", linewidths=1.5
        )
        nx.draw_networkx_labels(self.graph, pos, font_size=9, font_weight="bold")

        nx.draw_networkx_edges(
            self.graph,
            pos,
            arrows=True,
            arrowstyle="-|>",
            arrowsize=16,
            edge_color="#3182bd",
            connectionstyle="arc3,rad=0.12",
            min_source_margin=15,
            min_target_margin=15,
        )

        plt.axis("off")
        plt.tight_layout()
        plt.show()

    def draw_v2(self, save_path):
        G = self.graph
        labels = {node: node for node in G.nodes()}

        if nx.is_directed_acyclic_graph(G):
            order = list(nx.topological_sort(G))
        else:
            raise ValueError("The graph contains cycles! ")

        level = {}
        for node in order:
            preds = list(G.predecessors(node))
            if not preds:
                level[node] = 0
            else:
                level[node] = max(level[p] + 1 for p in preds)

        layers = {}
        for node, lev in level.items():
            layers.setdefault(lev, []).append(node)

        pos = {}
        x_gap = 2
        for lev, nodes_in_layer in layers.items():
            n = len(nodes_in_layer)
            y_positions = list(range(-(n - 1), n, 2))
            for i, node in enumerate(sorted(nodes_in_layer)):
                pos[node] = (lev * x_gap, y_positions[i])

        plt.figure(figsize=(12, 6))

        nx.draw_networkx_nodes(G, pos, node_color="lightblue", node_size=1500)
        nx.draw_networkx_labels(G, pos, labels, font_size=12)
        nx.draw_networkx_edges(
            G, pos,
            connectionstyle="arc3,rad=0.25",
            arrowstyle="-|>",
            arrowsize=20,
            edge_color="gray",
            min_source_margin=15,
            min_target_margin=15
        )
        plt.margins(0.3)
        plt.autoscale(enable=True)
        plt.axis("off")
        # plt.show()
        plt.savefig(save_path, dpi=300)
