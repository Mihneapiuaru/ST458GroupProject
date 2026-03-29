#!/usr/bin/env python3
"""
Compatibility wrapper for walk_forward.py.

The production strategy now lives in pca_kelly_strategy.py.
"""

from pca_kelly_strategy import State, initialise_state, trading_algorithm

__all__ = ["State", "initialise_state", "trading_algorithm"]

