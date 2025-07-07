"""
Consciousness Module - Core consciousness development system

This module contains the main consciousness development engine that orchestrates
humor generation, meta-cognitive reflection, and recursive self-improvement.
"""

from .meta_cognitive import MetaCognitiveEngine
from .theory_engine import TheoryEvolutionEngine
from .monitoring import ConsciousnessMonitor
from .metrics import ConsciousnessMetrics

# Main system class will be imported from a system.py file
try:
    from .system import ConsciousnessSystem
except ImportError:
    # For now, we'll create a placeholder
    ConsciousnessSystem = None

__all__ = [
    "ConsciousnessSystem",
    "MetaCognitiveEngine",
    "TheoryEvolutionEngine", 
    "ConsciousnessMonitor",
    "ConsciousnessMetrics",
] 