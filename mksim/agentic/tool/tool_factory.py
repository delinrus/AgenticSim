from mksim.agentic.tool.tools import *


class ToolFactory:
    @classmethod
    def create_tool(cls, tool_type, **kwargs):
        tool_class = cls._get_class(tool_type)
        if tool_class:
            return tool_class(**kwargs)
        raise ValueError(f"Unknown tool type: {tool_type}")

    @classmethod
    def _get_class(cls, task_type):
        task_map = {
            "embedding": EmbeddingTool,
            "infer": InferTool,
            "question": QuestionTool,
            "webtool": WebTool
        }
        return task_map.get(task_type)