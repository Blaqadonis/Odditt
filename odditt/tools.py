"""The CALC[] math tool: a restricted, safe expression evaluator (not raw eval()).

Extracted verbatim from the notebook's Section 6 cell.
"""
import ast
import operator
import re
from datetime import date

_ALLOWED_BINOPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod, ast.Pow: operator.pow,
}
_ALLOWED_UNARYOPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def _to_ordinal(date_str: str) -> int:
    # Whitelisted helper: converts a 'YYYY-MM-DD' string into a plain integer (days since a fixed
    # epoch, via Python's proleptic Gregorian calendar). This is the ONE thing a date needs that a
    # plain arithmetic expression can't do on its own -- calendars have irregular month lengths and
    # leap years, so "how many days between two dates" isn't expressible with +/-/*// alone without
    # first turning each date into a comparable number. Once that's done, subtraction, division,
    # rounding, etc. are all ordinary arithmetic the model writes itself in its own CALC[] expression
    # -- this function does NOT compute a difference, a week count, or anything else on your behalf.
    return date.fromisoformat(date_str).toordinal()


_ALLOWED_FUNCS = {
    "abs": abs, "round": round, "min": min, "max": max, "sum": sum,
    "to_ordinal": _to_ordinal,
}


class _SafeMathError(Exception):
    pass


def _safe_eval_node(node):
    if isinstance(node, ast.Expression):
        return _safe_eval_node(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float, str)):
            return node.value
        raise _SafeMathError(f"Unsupported constant type: {type(node.value)}")
    if isinstance(node, (ast.List, ast.Tuple)):
        return [_safe_eval_node(el) for el in node.elts]
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_BINOPS:
        return _ALLOWED_BINOPS[type(node.op)](_safe_eval_node(node.left), _safe_eval_node(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_UNARYOPS:
        return _ALLOWED_UNARYOPS[type(node.op)](_safe_eval_node(node.operand))
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in _ALLOWED_FUNCS:
            raise _SafeMathError("Only whitelisted functions may be called")
        args = [_safe_eval_node(a) for a in node.args]
        return _ALLOWED_FUNCS[node.func.id](*args)
    raise _SafeMathError(f"Unsupported expression node: {type(node).__name__}")


def safe_calculate(expression: str):
    """Evaluate a restricted arithmetic expression. Raises _SafeMathError on anything disallowed."""
    tree = ast.parse(expression, mode="eval")
    return _safe_eval_node(tree)


CALC_PATTERN = re.compile(r"CALC\[(.*?)\]")
