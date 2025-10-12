from abc import ABC, abstractmethod


class AgenticTool(ABC):
    def __init__(self, name: str, config):
        self.name = name
        self.config = config

    @abstractmethod
    def get_network_load(self):
        pass

    @abstractmethod
    def get_cpu_load(self):
        pass

    @abstractmethod
    def get_npu_load(self):
        pass

    @abstractmethod
    def get_memory_load(self):
        pass

    @abstractmethod
    def get_disk_load(self):
        pass

    def get_loads(self):
        return self.get_network_load(), self.get_cpu_load(), self.get_memory_load(), self.get_disk_load(), self.get_npu_load()


class WebTool(AgenticTool):

    def __init__(self, tool_name, config):
        super().__init__(tool_name, config)
        self.input_tokens = self.config.get('input_tokens', 0)
        self.output_tokens = self.config.get('output_tokens', 0)
        self.extraction_ratio = self.config.get('extraction_ratio', 1)
        self.bpe_token_size = self.config.get('bpe_token_size', 0)
        self.input_bytes = self.input_tokens * self.bpe_token_size
        self.output_bytes = self.output_tokens * self.bpe_token_size / self.extraction_ratio
        pass

    def get_network_load(self):
        return self.input_bytes + self.output_bytes

    def get_cpu_load(self):
        return 0

    def get_npu_load(self):
        return 0

    def get_memory_load(self):
        return self.input_bytes + self.output_bytes

    def get_disk_load(self):
        return 0


class InferTool(AgenticTool):

    def __init__(self, tool_name, config):
        super().__init__(tool_name, config)
        pass

    def get_network_load(self):
        return 0

    def get_cpu_load(self):
        return 0

    def get_npu_load(self):
        return 0

    def get_memory_load(self):
        return 0

    def get_disk_load(self):
        return 0


class EmbeddingTool(AgenticTool):

    def __init__(self, tool_name, config):
        super().__init__(tool_name, config)
        pass

    def get_network_load(self):
        return 0

    def get_cpu_load(self):
        return 0

    def get_npu_load(self):
        return 0

    def get_memory_load(self):
        return 0

    def get_disk_load(self):
        return 0


class QuestionTool(AgenticTool):

    def __init__(self, tool_name, config):
        super().__init__(tool_name, config)
        pass

    def get_network_load(self):
        return 0

    def get_cpu_load(self):
        return 0

    def get_npu_load(self):
        return 0

    def get_memory_load(self):
        return 0

    def get_disk_load(self):
        return 0

