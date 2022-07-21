"""Utilities for converting and operating on ONNX, JIT and torch types."""
from __future__ import annotations

import torch
from torch._C import _onnx as _C_onnx

from typing_extensions import Literal
from typing import Union

# Metaprogram symbolics for each ATen native specialized cast operator.
# For e.g. we specify a function named `_cast_uint8_t` that instantiates an
# ONNX cast node with `to` attribute "UINT8"
#
# TODO: remove these once we support Type's in the JIT IR and we can once again
# use the unified toType operator
TorchTypeName = Literal[
    "Byte",
    "Char",
    "Double",
    "Float",
    "Half",
    "Int",
    "Long",
    "Short",
    "Bool",
    "ComplexFloat",
    "ComplexDouble",
    "QInt8",
    "QUInt8",
    "QInt32",
    "BFloat16",
    "Undefined",
]

ScalarName = Literal[
    "uint8_t",
    "int8_t",
    "double",
    "float",
    "half",
    "int",
    "int64_t",
    "int16_t",
    "bool",
    "complex64",
    "complex128",
    "qint8",
    "quint8",
    "qint32",
    "bfloat16",
]

TORCH_TO_ONNX = {
    "Byte": _C_onnx.TensorProtoDataType.UINT8,
    "Char": _C_onnx.TensorProtoDataType.INT8,
    "Double": _C_onnx.TensorProtoDataType.DOUBLE,
    "Float": _C_onnx.TensorProtoDataType.FLOAT,
    "Half": _C_onnx.TensorProtoDataType.FLOAT16,
    "Int": _C_onnx.TensorProtoDataType.INT32,
    "Long": _C_onnx.TensorProtoDataType.INT64,
    "Short": _C_onnx.TensorProtoDataType.INT16,
    "Bool": _C_onnx.TensorProtoDataType.BOOL,
    "ComplexFloat": _C_onnx.TensorProtoDataType.COMPLEX64,
    "ComplexDouble": _C_onnx.TensorProtoDataType.COMPLEX128,
    "BFloat16": _C_onnx.TensorProtoDataType.BFLOAT16,
    "Undefined": _C_onnx.TensorProtoDataType.UNDEFINED,
}

# source of truth is
# https://github.com/pytorch/pytorch/blob/master/torch/csrc/utils/tensor_dtypes.cpp
TORCH_TO_DTYPE = {
    "Byte": torch.uint8,
    "Char": torch.int8,
    "Double": torch.double,
    "Float": torch.float,
    "Half": torch.half,
    "Int": torch.int,
    "Long": torch.int64,
    "Short": torch.short,
    "Bool": torch.bool,
    "ComplexFloat": torch.complex64,
    "ComplexDouble": torch.complex128,
    "QInt8": torch.qint8,
    "QUInt8": torch.quint8,
    "QInt32": torch.qint32,
    "BFloat16": torch.bfloat16,
}

# This indicates each scalar type's corresponding
# torch type. Related source:
# https://github.com/pytorch/pytorch/blob/344defc9733a45fee8d0c4d3f5530f631e823196/c10/core/ScalarType.h
SCALAR_NAME_TO_TORCH = {
    "uint8_t": "Byte",
    "int8_t": "Char",
    "double": "Double",
    "float": "Float",
    "half": "Half",
    "int": "Int",
    "int64_t": "Long",
    "int16_t": "Short",
    "bool": "Bool",
    "complex64": "ComplexFloat",
    "complex128": "ComplexDouble",
    "qint8": "QInt8",
    "quint8": "QUInt8",
    "qint32": "QInt32",
    "bfloat16": "BFloat16",
}


TORCH_TYPES = (
    torch.uint8,  # 0
    torch.int8,  # 1
    torch.short,  # 2
    torch.int,  # 3
    torch.int64,  # 4
    torch.half,  # 5
    torch.float,  # 6
    torch.double,  # 7
    torch.complex32,  # 8
    torch.complex64,  # 9
    torch.complex128,  # 10
    torch.bool,  # 11
    torch.qint8,  # 12
    torch.quint8,  # 13
    torch.qint32,  # 14
    torch.bfloat16,  # 15
)


def torch_to_onnx(torch_type: Union[TorchTypeName, str]) -> _C_onnx.TensorProtoDataType:
    return TORCH_TO_ONNX[torch_type]


def scalar_name_to_torch(scalar_name: Union[ScalarName, str]) -> TorchTypeName:
    return SCALAR_NAME_TO_TORCH[scalar_name]


def torch_to_dtype(torch_type: Union[TorchTypeName, str]) -> torch.dtype:
    return TORCH_TO_DTYPE[torch_type]


def valid_scalar_name(name: str) -> bool:
    return name in SCALAR_NAME_TO_TORCH


def onnx_compatible(torch_type: Union[TorchTypeName, str]) -> bool:
    return torch_type in TORCH_TO_ONNX


