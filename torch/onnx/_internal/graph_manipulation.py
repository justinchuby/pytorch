"""Utilities for manipulating torch graphs."""

from typing import Any, Dict, Iterable, Optional, Tuple, Union, Sequence, Mapping

import torch
from torch import _C
from torch._C import _onnx as _C_onnx

from torch.onnx import _constants


def _construct_opname(namespace: str, op: str) -> str:
    return f"{namespace}::{op}"


def _parse_opname(opname: str) -> Tuple[str, str]:
    namespace, op = opname.split("::")
    return namespace, op


def _new_node(
    g: _C.Graph, namespace: str, opname: str, outputs: int, *args, **kwargs
) -> _C.Node:
    """Creates a new node in the graph.

    Args:
        g: The graph to create the operator on.
        namespace: The namespace of the operator. E.g., "aten", "onnx".
        op: The name of the operator to create.
        outputs: The number of the outputs of the node.

    Returns:
        The new node.
    """
    aten = kwargs.pop("aten", False)
    node = g.create(f"{namespace}::{opname}", args, outputs)
    for k, v in sorted(kwargs.items()):
        if k == "inplace":
            continue
        _add_attribute(node, k, v, aten=aten)
    return node


def graph_create_op(
    graph: _C.Graph,
    namespace: str,
    opname: str,
    raw_args: Sequence[_C.Value],
    attributes: Mapping[str, Any],
    outputs: int = 1,
    onnx_infer_shapes: bool = True,
    onnx_opset_version: int = _constants.onnx_main_opset,
    global_params_dict: Optional[Dict[str, Any]] = None,
) -> Union[_C.Value, Tuple[_C.Value, ...]]:
    r"""Creates a TorchScript operator "namespace::opname".

    Args:
        graph: The Torch graph to create an op on.
        namespace: The namespace of the operator. E.g., "onnx".
        opname: The ONNX operator name, e.g., `Abs` or `Add`.
        raw_args: The inputs to the operator; usually provided
            as arguments to the `symbolic` definition.
        attributes: The attributes of the operator, whose keys are named
            according to the following convention: `alpha_f` indicates
            the `alpha` attribute with type `f`.  The valid type specifiers are
            `f` (float), `i` (int), `s` (string) or `t` (Tensor).  An attribute
            specified with type float accepts either a single float, or a
            list of floats (e.g., you would say `dims_i` for a `dims` attribute
            that takes a list of integers).
        outputs: The number of outputs this operator returns.
            By default an operator is assumed to return a single output.
            If `outputs` is greater than one, this functions returns a tuple
            of output `Node`, representing each output of the operator
            in order.
        onnx_infer_shapes: Whether to infer the shapes of the output tensors.
        onnx_opset_version: The ONNX opset version to run shape inference on.
        global_params_dict: A global dictionary of parameters to be used in the
            that is accessed by c++ code for shape inference.

    Returns:
        The node representing the single output of this operator (see the `outputs`
        keyword argument for multi-return nodes).
    """
    # Filter out None attributes, this can be convenient client side because
    # now they can pass through None attributes, and have them not show up
    attributes = {k: v for k, v in attributes.items() if v is not None}

    args = [
        _const_if_tensor(
            graph, arg, onnx_infer_shapes, onnx_opset_version, global_params_dict
        )
        for arg in raw_args
    ]

    node = graph.insertNode(
        _new_node(graph, namespace, opname, outputs, *args, attributes)
    )

    if onnx_infer_shapes:
        if global_params_dict is None:
            raise ValueError(
                "'global_params_dict' must be provided when 'onnx_infer_shapes' is True"
            )
        _C._jit_pass_onnx_node_shape_type_inference(
            node, global_params_dict, onnx_opset_version
        )

    if outputs == 1:
        return node.output()
    return tuple(node.outputs())


def _const_if_tensor(
    graph: _C.Graph,
    arg: Union[torch.Tensor, _C.Value],
    onnx_infer_shapes: bool,
    onnx_opset_version: int,
    global_params_dict: Optional[dict],
):
    if arg is None:
        return arg
    if isinstance(arg, _C.Value):
        return arg
    return graph_create_op(
        graph,
        "onnx",
        "Constant",
        raw_args=[],
        attributes=dict(value_z=arg),
        onnx_infer_shapes=onnx_infer_shapes,
        onnx_opset_version=onnx_opset_version,
        global_params_dict=global_params_dict,
    )


def _is_onnx_list(value):
    return (
        not isinstance(value, torch._six.string_classes)
        and not isinstance(value, torch.Tensor)
        and isinstance(value, Iterable)
    )


def _scalar(x: torch.Tensor):
    """Convert a scalar tensor into a Python value."""
    assert x.numel() == 1
    return x[0]


def _is_caffe2_aten_fallback():
    return (
        GLOBALS.operator_export_type == _C_onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK
        and _C_onnx._CAFFE2_ATEN_FALLBACK
    )


def _add_attribute(node: _C.Node, key: str, value: Any, aten: bool):
    r"""Initializes the right attribute based on type of value."""
    m = _ATTR_PATTERN.match(key)
    if m is None:
        raise ValueError(
            f"Invalid attribute specifier '{key}' names "
            " must be suffixed with type, e.g. 'dim_i' or 'dims_i'"
        )
    name, kind = m.group(1), m.group(2)
    if _is_onnx_list(value):
        kind += "s"

    if aten and _is_caffe2_aten_fallback():
        if isinstance(value, torch.Tensor):
            # Caffe2 proto does not support tensor attribute.
            if value.numel() > 1:
                raise ValueError("Should not pass tensor attribute")
            value = _scalar(value)
            if isinstance(value, float):
                kind = "f"
            else:
                kind = "i"
    return getattr(node, f"{kind}_")(name, value)
