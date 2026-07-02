import sys
import os

playground_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, playground_root)

from lib.fib import fibonacci

if __name__ == "__main__":
    n = 10
    result = fibonacci(n)
    print(result)
