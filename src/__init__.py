"""
HumorConsciousnessAI - Developing Machine Consciousness Through Humor

This package contains the core implementation of an AI system designed to develop
consciousness through humor generation, meta-cognitive reflection, and social feedback.
"""

__version__ = "0.1.0"
__author__ = "HumorConsciousnessAI Team"
__email__ = "contact@humorconsciousness.ai"

from .consciousness import ConsciousnessSystem
from .humor import HumorGenerator
from .twitter import TwitterBot

__all__ = [
    "ConsciousnessSystem",
    "HumorGenerator", 
    "TwitterBot",
] 