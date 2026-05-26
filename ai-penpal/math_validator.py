"""
math_validator.py
-----------------
Safe helpers for detecting, normalizing, and evaluating OCR-extracted
arithmetic expressions.
"""

import ast
import re

MATH_OPERATOR_CHARS = set("0123456789+-*/=()?.xX×÷−–— ")


def detect_math_expression(text: str) -> bool:
    return _extract_math_candidate(text) is not None


def normalize_math_expression(text: str) -> str:
    candidate = _extract_math_candidate(text) or text.strip()
    expression = candidate.strip()

    for original, replacement in (
        ("×", "*"),
        ("÷", "/"),
        ("−", "-"),
        ("–", "-"),
        ("—", "-"),
    ):
        expression = expression.replace(original, replacement)

    expression = re.sub(r"(?<=[\d)])\s*[xX]\s*(?=[\d(])", "*", expression)
    expression = expression.replace("[", "(").replace("]", ")").replace("{", "(").replace("}", ")")

    if "=" in expression:
        left_side, right_side = expression.split("=", 1)
        if not right_side.strip() or "?" in right_side:
            expression = left_side

    expression = expression.replace("?", "")
    expression = re.sub(r"\s+", "", expression)

    # Normalize repeated operator variants conservatively.
    expression = re.sub(r"(?<=\d|\))(?=\()", "*", expression)
    expression = re.sub(r"(?<=\))(?=\d)", "*", expression)

    return expression.strip()


def is_expression_suspicious(expression: str) -> bool:
    normalized_expression = normalize_math_expression(expression)
    if not normalized_expression:
        return True

    if normalized_expression.count("(") != normalized_expression.count(")"):
        return True

    if re.search(r"[^\d+\-*/().]", normalized_expression):
        return True

    numbers = re.findall(r"\d+", normalized_expression)
    parenthetical_groups = re.findall(r"\(([^()]*)\)", normalized_expression)

    if len(numbers) <= 4 and len(normalized_expression) <= 16:
        for group in parenthetical_groups:
            group_numbers = re.findall(r"\d+", group)
            if any(len(number) >= 2 for number in group_numbers):
                if sum(len(number) >= 2 for number in numbers) == 1:
                    return True

    # Expressions like 6+2*(14+2) are suspicious for OCR because the leading
    # operator and grouped number commonly come from a misread division pattern.
    if re.match(r"^\d+\+\d+\*\([^()]*\d{2,}[^()]*\)$", normalized_expression):
        return True

    if "(" in normalized_expression and ")" in normalized_expression and "*" in normalized_expression and "/" not in normalized_expression:
        prefix = normalized_expression.split("*", 1)[0]
        if "+" in prefix or "-" in prefix:
            return True

    return False


def safe_evaluate_expression(expression: str) -> dict:
    detected = detect_math_expression(expression)
    if not detected:
        return {
            "detected": False,
            "expression": "",
            "result": None,
            "error": None,
        }

    normalized_expression = normalize_math_expression(expression)
    if not normalized_expression:
        return {
            "detected": True,
            "expression": "",
            "result": None,
            "error": "No valid arithmetic expression found.",
        }

    try:
        tree = ast.parse(normalized_expression, mode="eval")
        result = _evaluate_ast(tree)
        return {
            "detected": True,
            "expression": normalized_expression,
            "result": _format_number(result),
            "error": None,
        }
    except ZeroDivisionError:
        return {
            "detected": True,
            "expression": normalized_expression,
            "result": None,
            "error": "Division by zero is not allowed.",
        }
    except Exception:
        return {
            "detected": True,
            "expression": normalized_expression,
            "result": None,
            "error": "Expression could not be evaluated safely.",
        }


def _extract_math_candidate(text: str) -> str | None:
    candidates = [line.strip() for line in text.splitlines() if line.strip()]
    if not candidates and text.strip():
        candidates = [text.strip()]

    best_candidate = None
    best_score = 0.0

    for candidate in candidates:
        score = _math_candidate_score(candidate)
        if score > best_score:
            best_candidate = candidate
            best_score = score

    if best_score >= 0.6:
        return best_candidate
    return None


def _math_candidate_score(candidate: str) -> float:
    if not candidate:
        return 0.0

    allowed_char_count = sum(1 for char in candidate if char in MATH_OPERATOR_CHARS)
    digit_groups = re.findall(r"\d+(?:\.\d+)?", candidate)
    operator_count = sum(1 for char in candidate if char in "+-*/=xX×÷−–—")

    if len(digit_groups) < 2 or operator_count == 0:
        return 0.0

    return allowed_char_count / max(len(candidate), 1)


def _evaluate_ast(tree: ast.AST):
    def evaluate(node):
        if isinstance(node, ast.Expression):
            return evaluate(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp):
            left = evaluate(node.left)
            right = evaluate(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                return left / right
            raise ValueError("Unsupported operator")
        if isinstance(node, ast.UnaryOp):
            operand = evaluate(node.operand)
            if isinstance(node.op, ast.UAdd):
                return +operand
            if isinstance(node.op, ast.USub):
                return -operand
            raise ValueError("Unsupported unary operator")
        raise ValueError("Unsupported expression node")

    return evaluate(tree)


def _format_number(value) -> str:
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return f"{value:.10g}"
