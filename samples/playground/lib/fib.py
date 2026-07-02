"""Fibonacci module."""


def fibonacci(n):
    """Iteratively return the nth Fibonacci number.

    Args:
        n: non-negative integer

    Returns:
        The nth Fibonacci number.

    Raises:
        ValueError: if n is negative.
    """
    if n < 0:
        raise ValueError("n must be non-negative")

    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a
