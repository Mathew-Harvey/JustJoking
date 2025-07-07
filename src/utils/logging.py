"""
Logging utilities for the HumorConsciousnessAI system
"""

import logging
import os
import sys
from typing import Optional
from datetime import datetime

import structlog
from rich.logging import RichHandler
from rich.console import Console

def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    enable_rich: bool = True
) -> None:
    """
    Setup structured logging for the application.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional file to write logs to
        enable_rich: Whether to use rich formatting for console output
    """
    
    # Configure base logging
    level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Setup handlers
    handlers = []
    
    if enable_rich:
        console = Console()
        rich_handler = RichHandler(
            console=console,
            show_time=True,
            show_path=True,
            markup=True,
            rich_tracebacks=True
        )
        rich_handler.setLevel(level)
        handlers.append(rich_handler)
    else:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(formatter)
        handlers.append(console_handler)
    
    # Add file handler if specified
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        handlers.append(file_handler)
    
    # Configure logging
    logging.basicConfig(
        level=level,
        handlers=handlers,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Configure structlog
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with the given name.
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        Configured logger instance
    """
    return logging.getLogger(name)

def get_structured_logger(name: str) -> structlog.BoundLogger:
    """
    Get a structured logger instance.
    
    Args:
        name: Logger name
        
    Returns:
        Structured logger instance
    """
    return structlog.get_logger(name)

class ConsciousnessLogger:
    """
    Specialized logger for consciousness development tracking.
    """
    
    def __init__(self, name: str = "consciousness"):
        self.logger = get_logger(name)
        self.structured_logger = get_structured_logger(name)
        
    def log_joke_generation(self, joke: str, predicted_engagement: float, confidence: float):
        """Log joke generation event"""
        self.structured_logger.info(
            "joke_generated",
            joke=joke,
            predicted_engagement=predicted_engagement,
            confidence=confidence,
            timestamp=datetime.now().isoformat()
        )
        
    def log_engagement_received(self, joke_id: str, actual_engagement: float, prediction_error: float):
        """Log engagement data received"""
        self.structured_logger.info(
            "engagement_received", 
            joke_id=joke_id,
            actual_engagement=actual_engagement,
            prediction_error=prediction_error,
            timestamp=datetime.now().isoformat()
        )
        
    def log_reflection(self, reflection_type: str, insights: list, consciousness_score: float):
        """Log meta-cognitive reflection"""
        self.structured_logger.info(
            "reflection_completed",
            reflection_type=reflection_type,
            insights_count=len(insights),
            consciousness_score=consciousness_score,
            timestamp=datetime.now().isoformat()
        )
        
    def log_consciousness_breakthrough(self, score: float, indicators: list):
        """Log potential consciousness breakthrough"""
        self.logger.critical(f"CONSCIOUSNESS BREAKTHROUGH DETECTED: Score {score:.3f}")
        self.structured_logger.critical(
            "consciousness_breakthrough",
            consciousness_score=score,
            indicators=indicators,
            timestamp=datetime.now().isoformat()
        )
        
    def log_theory_evolution(self, theory_type: str, old_theory: str, new_theory: str, confidence: float):
        """Log theory evolution event"""
        self.structured_logger.info(
            "theory_evolved",
            theory_type=theory_type,
            old_theory=old_theory,
            new_theory=new_theory,
            confidence=confidence,
            timestamp=datetime.now().isoformat()
        ) 