"""The base class for all ONNX operators."""
i

from typing import Any, List, Optional, Tuple, Union


class AttrType:
    FLOAT = "at::ScalarType::Float"
    FLOATS = "<unsupported:FLOATS>"
    INT = "at::ScalarType::Int"
    INTS = "<unsupported:INTS>"
    STRING = "const char*"
    STRINGS = "<unsupported:STRINGS>"
    TENSOR = "at::Tensor"
    LONG = "at::ScalarType::Long"


class ONNXAttr:
    def __init__(self, value, type: AttrType = None):
        self.value = value
        self.type = type


class OnnxOp:
    def __init__(
        self,
        name: str,
        outputs: int,
        input_types: List,
        inputs: str,
        attributes: Optional[str],
    ):
        # We don't care about the input type, because we assume it is going to be valid
        # we care about the argument types. They can be string, int or tensor.
        self.name = name
        self.outputs = outputs
        self.inputs = inputs
        self.attributes = attributes
        self.input_types = input_types

    def __call__(self, graph: _C.Graph) -> Any:
        pass

    def _create_node(self, graph: _C.Graph) -> _C.Node:
        return graph.create(self.name, self.inputs, self.attributes)
