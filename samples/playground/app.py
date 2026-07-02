import json  # seeded unused import for diagnostics tests
import math

try:
    from flask import Flask
except ImportError:
    class Flask:
        def __init__(self, name):
            self.name = name

        def route(self, rule):
            def decorator(f):
                f.rule = rule
                return f
            return decorator

        def add_url_rule(self, rule, endpoint=None, view_func=None, **kwargs):
            pass

        def run(self, **kwargs):
            pass


class Shape:
    def area(self):
        return 0


class Circle(Shape):
    def __init__(self, radius):
        self.radius = radius

    def area(self):
        return math.pi * self.radius ** 2


class Square(Shape):
    def __init__(self, side):
        self.side = side

    def area(self):
        return self.side ** 2


def create_app():
    app = Flask(__name__)

    circle = Circle(radius=3)
    square = Square(side=4)

    @app.route("/")
    def index():
        return {"circle_area": circle.area(), "square_area": square.area()}

    app.add_url_rule("/shapes", endpoint="shapes", view_func=lambda: {
        "circle": circle.area(),
        "square": square.area(),
    })

    return app
