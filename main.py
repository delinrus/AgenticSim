import yaml

from utils.agentic_tool_graph import AgenticToolGraph
from utils.config_iterator import extract_dependencies, tool_combinations_iterator

if __name__ == '__main__':
    new_task_graph_config = "./configs/config.yaml"

    with open(new_task_graph_config, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)

    if not config:
        exit()

    dependency_map = extract_dependencies(config)

    task_graph = None
    for tool_combination in tool_combinations_iterator(config):
        task_graph = AgenticToolGraph.build_from_objects(list(tool_combination), dependency_map)
    task_graph.draw()
