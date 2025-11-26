import ast
from typing import Dict, Any
from torchair.core.utils import logger
from torchair._ge_concrete_graph.ge_converter.converter_utils import *


def infer_ge_output_by_symbol_calculate(symbol_map, meta_output):
    parser = ExpressionParser(symbol_map)
    result = parser.parse_expression(str(meta_output))
    return result


class ExpressionParser:
    def __init__(self, symbol_map: Dict[str, Tensor]):
        """
        初始化表达式解析器

        Args:
            symbol_map: 符号到tensor对象的映射字典，如 {'s0': tensorX, 's1': tensorY}
        """
        self.symbol_map = symbol_map
        self.operator_map = {
            ast.Add: ge.Add,
            ast.Sub: ge.Sub,
            ast.Mult: ge.Mul,
            ast.Div: ge.Div,
            ast.Pow: ge.Pow,
        }

    def parse_expression(self, expression_str: str) -> Any:
        """
        解析字符串表达式并返回可执行的ge Api调用表达式

        Args:
            expression_str: 要解析的字符串表达式，如 "s0*s1", "s0**2 + s1"

        Returns:
            可执行的ge Api调用表达式
        """
        try:
            # 解析字符串为AST
            tree = ast.parse(expression_str, mode='eval')
            # 遍历AST并转换为ge Api调用表达式
            result = self._visit_node(tree.body)
            logger.debug(f"parse expression: {expression_str} result is: {result}")
            return result
        except Exception as e:
            raise ValueError(f"unable to parse expression: {e}")

    def _visit_node(self, node) -> Any:
        """递归遍历AST节点"""
        if isinstance(node, ast.Name):
            # 处理符号节点 (s0, s1, 等)
            symbol_name = node.id
            if symbol_name in self.symbol_map:
                return self.symbol_map[symbol_name]
            else:
                raise ValueError(f"undefined symbol: {symbol_name}")

        elif isinstance(node, ast.BinOp):
            # 处理二元操作符 (*, +, -, /, **, 等)
            left = self._visit_node(node.left)
            right = self._visit_node(node.right)

            op_type = type(node.op)
            if op_type in self.operator_map:
                ge_api = self.operator_map[op_type]
                return ge_api(left, right)
            else:
                raise ValueError(f"unsupported binOP: {op_type}")

        elif isinstance(node, ast.UnaryOp):
            # 处理一元操作符 (-, +, 等)
            operand = self._visit_node(node.operand)
            if isinstance(node.op, ast.USub):
                return ge.Neg(operand)
            elif isinstance(node.op, ast.UAdd):
                return operand  # +操作符不改变值
            else:
                raise ValueError(f"unsupported UnaryOp: {type(node.op)}")

        elif isinstance(node, ast.Constant):
            # 处理常量值
            return ge.Const(node.value, dtype=DataType.DT_INT64)

        elif isinstance(node, ast.Call):
            # 处理函数调用 (可选扩展)
            return self._visit_function_call(node)

        else:
            raise ValueError(f"unsupported AST node: {type(node)}")

    def _visit_function_call(self, node: ast.Call) -> Any:
        """处理函数调用（扩展功能）"""
        # 这里可以扩展支持更多函数，如 torch.sin, torch.cos 等
        func_name = node.func.id if isinstance(node.func, ast.Name) else ""
        args = [self._visit_node(arg) for arg in node.args]

        if hasattr(ge, func_name):
            return getattr(ge, func_name)(*args)
        else:
            raise ValueError(f"unsupported function: {func_name}")