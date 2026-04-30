from __future__ import annotations

import ast
import math

import torch


EPS = 1.0e-8
PHYSICAL_EPS = 1.0e-30
ALLOWED_FUNCTIONS = {
    "sin": torch.sin,
    "cos": torch.cos,
    "tan": torch.tan,
    "asin": torch.asin,
    "acos": torch.acos,
    "atan": torch.atan,
    "sinh": torch.sinh,
    "cosh": torch.cosh,
    "tanh": torch.tanh,
    "exp": torch.exp,
    "log": torch.log,
    "sqrt": torch.sqrt,
    "abs": torch.abs,
}
ALLOWED_BINOPS = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.BitAnd, ast.BitOr)
ALLOWED_UNARYOPS = (ast.UAdd, ast.USub)
ALLOWED_CMPOPS = (ast.Gt, ast.GtE, ast.Lt, ast.LtE, ast.Eq, ast.NotEq)


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def inverse_sigmoid(value: float) -> float:
    value = clamp(value, 1.0e-6, 1.0 - 1.0e-6)
    return math.log(value / (1.0 - value))


def validate_expression(node: ast.AST, allowed_names: set[str]) -> None:
    if isinstance(node, ast.Expression):
        validate_expression(node.body, allowed_names)
        return
    if isinstance(node, ast.BinOp):
        if not isinstance(node.op, ALLOWED_BINOPS):
            raise ValueError(f"Unsupported binary operator: {ast.dump(node.op)}")
        validate_expression(node.left, allowed_names)
        validate_expression(node.right, allowed_names)
        return
    if isinstance(node, ast.UnaryOp):
        if not isinstance(node.op, ALLOWED_UNARYOPS):
            raise ValueError(f"Unsupported unary operator: {ast.dump(node.op)}")
        validate_expression(node.operand, allowed_names)
        return
    if isinstance(node, ast.Compare):
        validate_expression(node.left, allowed_names)
        for operator in node.ops:
            if not isinstance(operator, ALLOWED_CMPOPS):
                raise ValueError(f"Unsupported comparison operator: {ast.dump(operator)}")
        for comparator in node.comparators:
            validate_expression(comparator, allowed_names)
        return
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in ALLOWED_FUNCTIONS:
            raise ValueError("Only direct calls to whitelisted math functions are allowed")
        if node.keywords:
            raise ValueError("Keyword arguments are not supported in target expressions")
        for argument in node.args:
            validate_expression(argument, allowed_names)
        return
    if isinstance(node, ast.Name):
        if node.id not in allowed_names:
            raise ValueError(f"Unknown symbol in expression: {node.id}")
        return
    if isinstance(node, ast.Constant):
        if not isinstance(node.value, (int, float, bool)):
            raise ValueError("Only numeric constants are allowed in target expressions")
        return
    raise ValueError(f"Unsupported syntax in target expression: {ast.dump(node)}")


def compile_expression(expression: str, allowed_names: set[str]):
    parsed = ast.parse(expression, mode="eval")
    validate_expression(parsed, allowed_names)
    return compile(parsed, "<function-target>", "eval")


def resolve_device(preference: str, device_ids: list[int] | None = None) -> torch.device:
    if preference == "cuda" and torch.cuda.is_available():
        cuda_device_count = torch.cuda.device_count()
        requested_ids = [int(device_id) for device_id in (device_ids or [])]
        if requested_ids:
            for device_id in requested_ids:
                if 0 <= device_id < cuda_device_count:
                    return torch.device(f"cuda:{device_id}")
            raise ValueError(
                f"None of device_ids={requested_ids} are valid for {cuda_device_count} visible CUDA device(s)"
            )
        return torch.device("cuda:0")
    return torch.device("cpu")