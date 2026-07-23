"""Serving package for the car price predictor.

Note: the CD deploy step uploads this folder to a HF Space with ``__init__.py`` excluded,
so on the Space the modules import each other flatly (``import predictor``) while locally
they form the ``serving`` package (``serving.predictor``). See ``app.py`` for the bridge.
"""

__version__ = "0.1.0"
