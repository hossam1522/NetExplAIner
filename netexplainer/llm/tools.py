"""External tools that can be bound to the LLM (function calling)."""

import logging
import math

import numexpr
from langchain_core.tools import tool

logger = logging.getLogger(__name__)


@tool
def calculator(expression: str) -> str:
    """Calculate expression using Python's numexpr library.

    Expression should be a single line mathematical expression
    that solves the problem.

    Examples:
        "37593 * 67" for "37593 times 67"
        "37593**(1/5)" for "37593^(1/5)"
    """
    logger.debug(f"Calculator tool called with expression: {expression}")
    local_dict = {"pi": math.pi, "e": math.e}
    return str(
        numexpr.evaluate(
            expression.strip(),
            global_dict={},  # restrict access to globals
            local_dict=local_dict,  # add common mathematical functions
        )
    )


TOOLS = [calculator]
