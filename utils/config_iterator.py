import copy
from itertools import product
from typing import Iterator, Dict, Any, Tuple

from mksim.agentic.tool.tool_factory import ToolFactory
from mksim.agentic.tool.tools import AgenticTool


def toll_config_iterator(cfg: Dict[str, Any]) -> Iterator[Tuple[str, str, dict, dict]]:
    agents = cfg.get("Agents", {})
    if not isinstance(agents, dict):
        raise TypeError("cfg['Agents'] must be a mapping of stages to their configs")

    for stage_name, stage_conf in agents.items():
        stage_conf = stage_conf or {}
        tools = stage_conf.get("tools", {}) or {}
        if not isinstance(tools, dict):
            raise TypeError(f"Agents['{stage_name}']['tools'] must be a mapping of task_name -> task_params")

        for task_name, task_params in tools.items():
            task_params = task_params or {}
            yield stage_name, task_name, task_params, stage_conf


def single_tool_variations_iterator(tool_name: str, tool_params: dict) -> Iterator[AgenticTool]:
    variable_fields = tool_params.get('variable_fields', [])
    if variable_fields:
        parameter_list = [tool_params[field_name] for field_name in variable_fields]
        for combo in product(*parameter_list):
            tool_param_copy = copy.deepcopy(tool_params)
            for i, field_name in enumerate(variable_fields):
                tool_param_copy[field_name] = combo[i]

            tool_obj = ToolFactory.create_tool(
                tool_name=tool_name,
                tool_type = tool_param_copy.get("task_type"),
                config=tool_param_copy
            )
            yield tool_obj
    else:
        tool_obj = ToolFactory.create_tool(
            tool_name=tool_name,
            tool_type= tool_params.get("task_type"),
            config=tool_params
        )
        yield tool_obj


def tool_combinations_iterator(cfg: Dict[str, Any]) -> Iterator[Tuple[AgenticTool]]:
    tool_iterators = []
    for _, tool_name, tool_params, _ in toll_config_iterator(cfg):
        tool_iterators.append(single_tool_variations_iterator(tool_name, tool_params))
    return product(*tool_iterators)


def extract_dependencies(cfg: Dict[str, Any]):
    res = {}
    for _, tool_name, tool_params, _ in toll_config_iterator(cfg):
        res[tool_name] = tool_params['dependencies']
    return res

