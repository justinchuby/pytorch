"""Type promotion rules applied to aten operators."""

import abc
import torch
import logging
from typing import Any, TYPE_CHECKING
from torch import _prims_common, _refs

if TYPE_CHECKING:
    import onnxscript.values
    from onnxscript import ir


logger = logging.getLogger(__name__)



def _get_higher_dtype(a: ir.DataType, b: ir.DataType) -> ir.DataType:
    if a == b:
        return a

    if a is None:
        return b

    if b is None:
        return a

    ordered_datatypes = (
        (ir.DataType.BOOL,),
        (ir.DataType.UINT8, ir.DataType.INT8),
        (ir.DataType.INT16,),
        (ir.DataType.INT32,),
        (ir.DataType.INT64,),
        (ir.DataType.FLOAT16, ir.DataType.BFLOAT16),
        (ir.DataType.FLOAT,),
        (ir.DataType.DOUBLE,),
        (ir.DataType.COMPLEX64,),
        (ir.DataType.COMPLEX128,),
    )

    for idx, dtypes in enumerate(ordered_datatypes):
        if a in dtypes and b in dtypes:
            return ordered_datatypes[idx + 1][0]
        if a in dtypes:
            return b
        if b in dtypes:
            return a


def _promote_types(*onnx_dtypes: ir.DataType) -> ir.DataType:
    """Promote the types of the given ONNX data types according to PyTorch Rules

    Args:
        onnx_dtypes: ONNX data types to promote.

    Returns:
        Promoted ONNX data type.
    """
    if len(onnx_dtypes) == 1:
        return onnx_dtypes[0]

    promoted_dtype = onnx_dtypes[0]
    for dtype in onnx_dtypes[1:]:
        promoted_dtype = _get_higher_dtype(promoted_dtype, dtype)

    return promoted_dtype



class TypePromotionRule(abc.ABC):
    """Base class for type promotion rule per 'torch.ops.{namespace}.{op_name}'."""


    def __init__(self, target: torch._ops.OpOverloadPacket):
        self.target = target

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.target})"

    @abc.abstractmethod
    def promote_types(
        self, op: onnxscript.values.Opset, args: tuple[ir.Value | float | int | bool, ...]
    ) -> tuple[tuple[Any], dict[str, Any]]:
        """Promote the types of the arguments and keyword arguments for the operator."""

class ElementwiseTypePromotionRule(TypePromotionRule):
    """Type promotion rule for elementwise operators."""

    def __init__(
        self,
        target: torch._ops.OpOverloadPacket,
        promote_args_positions: tuple[int, ...],
        promotion_kind: _prims_common.ELEMENTWISE_TYPE_PROMOTION_KIND,
    ):
        """Constructs a TypePromotionRule for elementwise operators.

        Args:
            namespace: Namespace of the op. E.g. 'aten' in 'torch.ops.aten.add'.
            op_name: Name of the op. E.g. 'add' in 'torch.ops.aten.add'.
            promote_args_positions: Positions of args to promote.
            promote_kwargs_names: Names of kwargs to promote.
            promotion_kind: Type promotion kind. Refer to [_prims_common.elementwise_dtypes](https://github.com/pytorch/pytorch/blob/main/torch/_prims_common/__init__.py) for detail.  # noqa: B950
        """
        super().__init__(target)
        self.promote_args_positions = promote_args_positions
        self.promotion_kind = promotion_kind

    def promote_types(
        self, op: onnxscript.values.Opset, args: tuple[ir.Value | float | int | bool, ...]
    ) -> tuple[ir.Value | float | int | bool, ...]:
        promoted_args = list(args)
        promoted_types = []
        for idx in self.promote_args_positions:
            promoted_types.append(promoted_args[idx].dtype)
        promoted_dtype = _promote_types(*promoted_types)
        for idx in self.promote_args_positions:
            promoted_args[idx] = op.Cast(promoted_args[idx], promoted_dtype)
        return tuple(promoted_args), {}