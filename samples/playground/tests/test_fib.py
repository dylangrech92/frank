from lib.fib import fibonacci


def test_fibonacci_base_cases():
    assert fibonacci(0) == 0
    assert fibonacci(1) == 1


def test_fibonacci_ten():  # deliberately failing test for the test-explorer live tests
    assert fibonacci(10) == 54
